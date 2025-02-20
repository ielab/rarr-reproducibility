import gc
import glob
import json
import logging
from itertools import chain
from typing import List, Union, Dict, Iterable, Tuple
from tqdm import tqdm
import heapq
import time

import numpy as np
from numpy import ndarray
import torch
import faiss

from pyserini.search.lucene import LuceneSearcher
from retrieval.models import StellarEncoder, JinaEncoder, BM25Searcher
from retrieval.dataset import WikiChunkedDataset
from retrieval.hybrid import fuse
from retrieval.utils import pickle_load, write_ranking


logger = logging.getLogger(__name__)


def combine_faiss_results(results: Iterable[Tuple[ndarray, ndarray]],
                          num_query: int):
    all_scores_indices = [[] for _ in range(num_query)]

    # Iterate over the results and add them to the heap
    for scores, indices in results:
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                heapq.heappush(all_scores_indices[i], (-scores[i, j], indices[i, j]))

    # Extract the results back into arrays
    combined_scores = []
    combined_indices = []

    for i in range(num_query):
        scores_indices = all_scores_indices[i]
        scores = []
        indices = []
        while scores_indices:
            score, index = heapq.heappop(scores_indices)
            scores.append(-score)
            indices.append(str(index))
        combined_scores.append(scores)
        combined_indices.append(indices)
    return combined_scores, combined_indices



def load_shards(document_rep_files):
    """
    Loads multiple pickle files containing (p_reps, p_lookup) pairs,
    adds all passage representations to a FaissFlatSearcher,
    and returns the retriever, the concatenated lookup list, and a shard-index mapping.

    Args:
        document_rep_files (list): A list of file paths containing document representations
                                   and lookup data. Each file should contain a tuple (p_reps, p_lookup).

    Returns:
        tuple: (retriever, lookup, shard_map) where:
               - retriever is an instance of FaissFlatSearcher with all representations added.
               - lookup is a list mapping internal indices to document IDs.
               - shard_map is a dict mapping document IDs to shard identifiers.
    """

    # Get shard indices from the name. They are in the following format:

    # Determine shard IDs from filenames (e.g., file.corpus.0.pkl, file.corpus.12.pkl, ...)
    shard_ids = [int(file.split('.')[-2]) for file in document_rep_files]

    # Load the first shard to initialize the retriever.
    p_reps_0, p_lookup_0 = pickle_load(document_rep_files[0])

    # instantiate FAISS searcher with first set of embeddings
    document_vector_index = FaissFlatSearcher(p_reps_0)

    # load subsequent shards
    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, document_rep_files[1:]))
    if len(document_rep_files) > 1:
        from tqdm import tqdm
        shards = tqdm(shards, desc='Loading shards into index', total=len(document_rep_files))

    lookup = []
    shard_map = {}
    for shard_id, (p_reps, p_lookup) in zip(shard_ids, shards):
        document_vector_index.add(p_reps)
        lookup += p_lookup
        shard_map.update({doc: shard_id for doc in p_lookup})

    return document_vector_index, lookup, shard_map


class FaissFlatSearcher:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool = False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices

    def move_to_gpus(self):
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            logger.info("No GPU found. Back to CPU.")
        else:
            logger.info(f"Using {num_gpus} GPU")
            if num_gpus == 1:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co,
                                                         ngpu=num_gpus)


class FaissSearcher(FaissFlatSearcher):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)


class WikiSearcher:
    def __init__(self,
                 dataset_cache_dir: str,
                 retrieval_model_name: str,
                 document_reps_dir: str = None,
                 model_cache_dir: str = None,
                 use_gpu: bool = False,
                 torch_dtype: str = 'fp32',
                 query_prompt: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ",
                 debug: bool = False):
        """
        Initializes the WikiSearcher.

        Args:
            document_reps_dir (str): Glob pattern for document representation files.
            dataset_cache_dir (str): Directory for dataset cache.
            retrieval_model_name: The name of the retreival model {"stella_1.5B", "jina_v2_base", "bm25"}
            model_cache_dir (str, optional): Directory for model cache.
            use_gpu (bool): Whether to use GPU.
            torch_dtype (str): The torch data type to use.
            query_prompt (str): The prompt to use when encoding queries.
            debug (bool): Debug mode flag.
        """
        self.debug = debug
        self.dataset_cache_dir = dataset_cache_dir
        self.retrieval_model_name = retrieval_model_name.lower()

        # set the torch data type
        if torch_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif torch_dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        encoder_model_names = ["stella", "jina"]

        # decide what to do based on the retrieval model name.
        if self.retrieval_model_name in encoder_model_names:
            print("using encoder based retrieval model")
            # for encoder-based models, load document embeddings.
            self.document_rep_files = sorted(glob.glob(document_reps_dir))
            if self.debug:
                print("Debug mode: using a subset of document source files")
                self.document_rep_files = self.document_rep_files[:2]
            logger.info(f'Found {len(self.document_rep_files)} document representation files; loading into index.')
            self.document_vector_index, self.look_up, self.shard_map = load_shards(self.document_rep_files)
            # Instantiate the encoder-based retrieval backend.
            if self.retrieval_model_name == "stella":
                from retrieval.models import StellarEncoder
                self._retrieval_backend = StellarEncoder(model_cache_dir, torch_dtype, query_prompt)
            elif self.retrieval_model_name == "jina":
                from retrieval.models import JinaEncoder
                self._retrieval_backend = JinaEncoder(model_cache_dir, torch_dtype)
        elif self.retrieval_model_name == "bm25":
            print("using bm25 based retrieval model")
            # For BM25, no document embeddings are needed.
            self.document_rep_files = None
            self.document_vector_index = None
            self.look_up = None
            self.shard_map = None
            from retrieval.models import BM25Searcher
            lucene_index_dir = "/scratch/project/neural_ir/jj/datasets/wikipedia_20223_chunked/bm_25_index"
            self._retrieval_backend = BM25Searcher(lucene_index_dir)
        else:
            raise ValueError(f"Unknown retrieval model: {retrieval_model_name}")

        self.use_gpu = use_gpu
        if use_gpu and self.document_vector_index is not None:
            self.document_vector_index.move_to_gpus()

    def encode_queries(self, queries):
        """Encode queries using the encoder-based retrieval backend."""
        with torch.no_grad():
            query_reps = self._retrieval_backend(queries, encode_is_query=True)
        return query_reps.cpu().float().numpy()


    def run_vector_search(self, query_reps, k):
        """Find top k passages based on similarity with queries."""
        retrieval_scores, index_passage_ids = self.document_vector_index.search(query_reps, k)
        corpus_passage_ids = [[str(self.look_up[doc]) for doc in qid] for qid in index_passage_ids]
        return np.array(corpus_passage_ids), retrieval_scores


    def retrieve_relevant_passages(self, queries, k):
        """
        (1) For BM25, perform a BM25 search directly on the raw queries.
        (2) For encoder-based retrieval, encode queries and perform a vector search.
        (3) Retrieve passage text from corpus.

        Returns:
            passage_texts_per_query: List of lists of passage texts.
            retrieval_scores: List of lists of retrieval scores.
        """

        t0 = time.time()

        # for BM25 skip encoding and vector search.
        if self.retrieval_model_name == "bm25":
            # directly search using BM25
            doc_ids, retrieval_scores = self._retrieval_backend.search(queries, k)
        else:
            # For encoder-based models, encode the queries first.
            query_reps = self.encode_queries(queries)
            t0_enc = time.time()
            print("Time taken to encode: ", t0_enc - t0)

            if self.debug:
                doc_ids, retrieval_scores = self._run_random_search(n_queries=len(queries), k=k)
            else:
                doc_ids, retrieval_scores = self.run_vector_search(query_reps, k)

        t1 = time.time()

        print("Time taken to search: ", t1 - t0)

        passage_texts_per_query = self.retrieve_passage_texts_by_id(doc_ids)
        t2 = time.time()
        print("Time taken to retrieve text: ", t2 - t1)

        return passage_texts_per_query, retrieval_scores

    def retrieve_passage_texts_by_id(self, corpus_passage_ids):
        if self.retrieval_model_name == "bm25":
            # For BM25, the Lucene index returns documents based on chunked_id.
            # We now convert each document into an evidence dictionary.
            evidence_list_per_query = []
            for ids in corpus_passage_ids:
                evidences = []
                for passage_id in ids:
                    doc = self._retrieval_backend.get_doc(passage_id)
                    doc_json = json.loads(doc.raw())
                    evidence = {
                        'id': doc_json.get('id', passage_id),
                        'url': doc_json.get('url', ''),
                        'title': doc_json.get('title', ''),
                        # Here we map the original "contents" field to "text"
                        'text': doc_json.get('contents', ''),
                        'chunked_id': passage_id
                    }
                    evidences.append(evidence)
                evidence_list_per_query.append(evidences)
            return evidence_list_per_query
        else:
            corpus_passage_ids_flattened = corpus_passage_ids.flatten()
            shard_indices = [self.shard_map[x] for x in corpus_passage_ids_flattened]
            unique_shards = set(shard_indices)

            #  Helper method that loads each shard and returns a dict {passage_id -> text}
            psg_text = self._load_and_retrieve_text(unique_shards,
                                                corpus_passage_ids_flattened,
                                                shard_indices)

            # Now rebuild the query structure
            passage_texts_per_query = []
            for q_idx in range(len(corpus_passage_ids)):
                texts_for_this_query = [
                    psg_text[p_id] for p_id in corpus_passage_ids[q_idx]
                ]
                passage_texts_per_query.append(texts_for_this_query)
            return passage_texts_per_query

    def _load_and_retrieve_text(self, unique_shards, flattened_ids, shard_indices):
        """
        Given the unique shards we need,
        plus the array of flattened IDs and their corresponding shard indices,
        load the datasets and return a dictionary of {passage_id -> text}.
        """
        psg_text = {}
        for shard_idx in unique_shards:
            dset = WikiChunkedDataset(
                dataset_cache_dir=self.dataset_cache_dir,
                dataset_shard_index=shard_idx
            )
            # IDs that belong to this shard
            shard_psg_indices = np.where(np.array(shard_indices) == shard_idx)
            shard_psg_ids = flattened_ids[shard_psg_indices]

            # Retrieve text from the dataset
            shard_psg_text = [dset.get_idem_by_id(idx) for idx in shard_psg_ids]

            # Populate the dictionary
            for idx, passage_text in zip(shard_psg_ids, shard_psg_text):
                psg_text[idx] = passage_text

            # Cleanup
            del dset
            gc.collect()

        return psg_text


    def _run_random_search(self, n_queries: int, k: int):
        """
        Return random passage IDs for each query.
        """
        # total number of passages
        num_total_passages = len(self.look_up)

        # store results
        random_passage_ids_all = []

        for _ in range(n_queries):
            # randomly choose 'k' indices
            random_indices = np.random.randint(0, num_total_passages, size=k)
            # map them to actual passage IDs (strings)
            random_passage_ids = [str(self.look_up[i]) for i in random_indices]
            random_passage_ids_all.append(random_passage_ids)

        # shape: (n_queries, k)
        corpus_passage_ids = np.array(random_passage_ids_all)
        # for scores, just return zeros or random placeholders
        retrieval_scores = np.zeros((n_queries, k))

        return corpus_passage_ids, retrieval_scores


class PubmedHybridSearcher:
    def __init__(self,
                 lucene_index_path: str,
                 document_reps: str,
                 use_gpu: bool = False,
                 # use gpu for faiss, for big index dont use gpu since no memory. The encoder model will auto use gpu if available
                 cache_dir: str = None,
                 torch_dtype: str = 'fp32',
                 query_prompt: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ",
                 reduce: bool = False):
        self.reduce = reduce
        self.use_gpu = use_gpu
        # self.lucene_searcher = LuceneSearcher(lucene_index_path)
        # self.lucene_searcher.set_bm25(k1=2.5, b=0.9)  # best hyperparameters for question + topic + narrative

        self.index_files = glob.glob(document_reps)

        if not self.reduce:
            logger.info(f'Pattern match found {len(self.index_files)} files; loading them into index.')
            self.retriever, self.look_up, _ = load_shards(self.index_files)
            if use_gpu:
                self.retriever.move_to_gpus()

        if torch_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif torch_dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.encoder = StellarEncoder(cache_dir,
                                      torch_dtype,
                                      query_prompt)

    def search(self, queries: List[Union[str, dict[str, str]]], k: int) -> dict:
        qids = [str(i) for i in range(len(queries))]

        if isinstance(queries[0], dict):
            bm25_queries = [f"{query['question']} {query['topic']} {query['narrative']}" for query in queries]
            dense_queries = [query['question'] for query in queries]
        else:
            bm25_queries = queries
            dense_queries = queries

        # bm25_hits = self.lucene_searcher.batch_search(bm25_queries, qids, k=k)
        # bm25_run = {}
        # for qid, hits in bm25_hits.items():
        #     bm25_run[qid] = {'docs': {hit.docid: hit.score for hit in hits},
        #                      'max_score': float(hits[0].score),
        #                      'min_score': float(hits[-1].score)}

        with torch.no_grad():
            query_reps = self.encoder(dense_queries, encode_is_query=True).cpu().float().numpy()

        if not self.reduce:
            all_scores, all_indices = self.retriever.search(query_reps, k)
        else:
            dense_hits = []
            for rep_file in tqdm(self.index_files, desc='dense retrieval'):
                retriever, p_lookup = load_shards([rep_file])
                scores, indices = retriever.search(query_reps, k)
                indices = np.array([[p_lookup[doc] for doc in qid] for qid in indices])
                dense_hits.append((scores, indices))
                del retriever
                torch.cuda.empty_cache()

            all_scores, all_indices = combine_faiss_results(dense_hits, len(queries))

        all_indices = [[str(self.look_up[doc]) for doc in qid] for qid in all_indices]
        dense_run = {}
        for qid, scores, indices in zip(qids, all_scores, all_indices):
            dense_run[qid] = {'docs': {doc: score for doc, score in zip(indices, scores)},
                              'max_score': float(scores[0]),
                              'min_score': float(scores[-1])}

        # hybrid_run = fuse([bm25_run, dense_run], [0.2, 0.8])
        hybrid_run = fuse([dense_run, dense_run], [0.5, 0.5])

        result = {}
        for qid in hybrid_run:
            result[qid] = []
            q_d_scores = [(doc, score) for doc, score in hybrid_run[qid].items()]
            # sort by score
            q_d_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(q_d_scores[:k]):
                text = json.loads(self.lucene_searcher.doc(docid).raw())['contents']
                result[qid].append({'docid': docid, 'rank': rank, 'score': score, 'text': text})

        return result