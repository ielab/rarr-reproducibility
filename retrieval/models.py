from transformers import AutoModel, AutoTokenizer
from torch import nn, Tensor, dtype
import torch
import os
from typing import List, Union
from pyserini.search.lucene import LuceneSearcher

def get_model(model_args, training_args):
    MODEL_DICT = {'dunzhang/stella_en_1.5B_v5': StellarEncoder,
                  'jinaai/jina-embeddings-v2-base-en': JinaEncoder}
    try:
        encoder_class = MODEL_DICT[model_args.model_name_or_path]
    except KeyError:
        raise ValueError(f'{training_args.hub_model_id} encoder is not implemented')

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = encoder_class(model_args.cache_dir, torch_dtype)
    return model


class EncoderModel(nn.Module):
    def __init__(self, cache_dir: str, torch_dtype: dtype, query_prompt: str):
        super().__init__()
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.tokenizer = None
        self.query_prompt = query_prompt

    def encode_document(self, input_data) -> Tensor:
        raise NotImplementedError(f'encode_document is not implemented for {self.__class__.__name__}')

    def encode_query(self, input_data) -> Tensor:
        raise NotImplementedError(f'encode_query is not implemented for {self.__class__.__name__}')

    def forward(self, texts: List[str], encode_is_query: bool) -> Tensor:
        raise NotImplementedError(f'forward is not implemented for {self.__class__.__name__}')


class StellarEncoder(EncoderModel):
    def __init__(self,
                 cache_dir: str,
                 torch_dtype: dtype = 'float32',
                 query_prompt: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "):
        super().__init__(cache_dir, torch_dtype, query_prompt)
        model_id = 'dunzhang/stella_en_1.5B_v5'
        from huggingface_hub import snapshot_download
        revision = "2ce8247a8bb6a9702f9d4d77a04c7a233d6e9447"
        model_path = snapshot_download(model_id, cache_dir=self.cache_dir, revision=revision)
        # check if cuda available
        has_cuda = torch.cuda.is_available()

         # cuda for linux machine
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # mps for apple silicon
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"device set to {self.device}")

        self.model = AutoModel.from_pretrained(model_id,
                                               trust_remote_code=True,
                                               cache_dir=self.cache_dir,
                                               torch_dtype=self.torch_dtype)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       trust_remote_code=True,
                                                       cache_dir=self.cache_dir)
        vector_dim = 1024
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.vector_linear = nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim,
                                       dtype=self.torch_dtype)

        vector_linear_directory = f"2_Dense_{vector_dim}"
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_path, f"{vector_linear_directory}/pytorch_model.bin"),
                       weights_only=True,
                       map_location=("cpu" if not has_cuda else None)).items()
        }

        self.vector_linear.load_state_dict(vector_linear_dict)

        self.vector_linear.to(self.model.device)

    def encode_document(self, input_data):
        attention_mask = input_data["attention_mask"]
        last_hidden_state = self.model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embeddings = nn.functional.normalize(self.vector_linear(embeddings), p=2, dim=1)
        return embeddings

    def encode_query(self, input_data):
        return self.encode_document(input_data)

    def tokenize_text(self, prompt: List[str]) -> dict:
        tokenized_prompt = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=32,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        tokenized_prompt = self.tokenizer.pad(
            tokenized_prompt,
            padding=True,
            pad_to_multiple_of=16,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return tokenized_prompt

    def forward(self, texts: List[str], encode_is_query=False) -> Tensor:
        if encode_is_query:
            texts = [self.query_prompt + text for text in texts]

        input_data = self.tokenizer(texts,
                                    padding="longest",
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt").to(self.model.device)

        return self.encode_document(input_data)

class JinaEncoder(EncoderModel):
    def __init__(self,
                 cache_dir: str,
                 torch_dtype: dtype = 'float32',
                 query_prompt: str = ""):
        super().__init__(cache_dir, torch_dtype, query_prompt)
        model_id = 'jinaai/jina-embeddings-v2-base-en'
        
        from huggingface_hub import snapshot_download
        revision = "322d4d7e2f35e84137961a65af894fda0385eb7a"
        model_path = snapshot_download(model_id, cache_dir=self.cache_dir, revision=revision)
        # check if cuda available
        has_cuda = torch.cuda.is_available()

         # cuda for linux machine
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # mps for apple silicon
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"device set to {self.device}")

        self.model = AutoModel.from_pretrained(model_id,
                                               trust_remote_code=True,
                                               cache_dir=self.cache_dir,
                                               torch_dtype=self.torch_dtype)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       trust_remote_code=True,
                                                       cache_dir=self.cache_dir)
        vector_dim = 768
        self.vector_linear = nn.Linear(in_features=self.model.config.hidden_size,
                                       out_features=vector_dim,
                                       dtype=self.torch_dtype)

        # load the entire checkpoint from the snapshot root
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"),
                                weights_only=True,
                                map_location="cpu")

        # the vector linear layer’s weights are stored in the checkpoint under a specific key,
        vector_linear_dict = {
            k.replace("pooler.dense.", ""): v
            for k, v in state_dict.items() if k.startswith("pooler.dense.")
        }
        self.vector_linear.load_state_dict(vector_linear_dict)

        self.vector_linear.load_state_dict(vector_linear_dict)

        self.vector_linear.to(self.model.device)

    def encode_document(self, input_data):
        attention_mask = input_data["attention_mask"]
        last_hidden_state = self.model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embeddings = nn.functional.normalize(self.vector_linear(embeddings), p=2, dim=1)
        return embeddings

    def encode_query(self, input_data):
        return self.encode_document(input_data)

    def tokenize_text(self, prompt: List[str]) -> dict:
        tokenized_prompt = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=32,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        tokenized_prompt = self.tokenizer.pad(
            tokenized_prompt,
            padding=True,
            pad_to_multiple_of=16,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return tokenized_prompt

    def forward(self, texts: List[str], encode_is_query=False) -> Tensor:
        if encode_is_query:
            texts = [self.query_prompt + text for text in texts]

        input_data = self.tokenizer(texts,
                                    padding="longest",
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt").to(self.model.device)

        return self.encode_document(input_data)


class BM25Searcher:
    """
    A BM25-based retrieval model that wraps a Lucene index via Pyserini.
    This class implements a simple search() method that takes in a query (or a list of queries)
    and returns a tuple (doc_ids, scores), where:
      - doc_ids is a list of lists of document IDs (strings), one per query.
      - scores is a list of lists of BM25 scores (floats), one per query.
    """

    def __init__(self, index_dir: str, k1: float = 0.9, b: float = 0.4):
        """
        Initializes the BM25Searcher.

        Args:
            index_dir (str): Path to the pre-built Lucene index.
            k1 (float): BM25 k1 parameter.
            b (float): BM25 b parameter.
        """
        self.searcher = LuceneSearcher(index_dir)
        # Set BM25 parameters if available. (Pyserini's LuceneSearcher provides set_bm25.)
        self.searcher.set_bm25(k1, b)

    def get_doc(self, doc_id):
        # Helper method to retrieve a document by its ID.
        return self.searcher.doc(doc_id)

    def search(self, queries: Union[str, List[str]], k: int):
        """
        Searches for the top k documents for each query using BM25.

        Args:
            queries (Union[str, List[str]]): A query string or list of query strings.
            k (int): Number of top documents to retrieve per query.

        Returns:
            tuple: (doc_ids, scores) where
                - doc_ids is a list of lists of document IDs (strings),
                - scores is a list of lists of BM25 scores (floats).
        """
        # Ensure queries is a list.
        if isinstance(queries, str):
            queries = [queries]

        all_doc_ids = []
        all_scores = []

        for query in queries:
            hits = self.searcher.search(query, k)
            # Extract document IDs and scores from the hits.
            doc_ids = [hit.docid for hit in hits]
            scores = [hit.score for hit in hits]
            all_doc_ids.append(doc_ids)
            all_scores.append(scores)

        return all_doc_ids, all_scores