import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from retrieval.utils import write_ranking, pickle_load

from retrieval.searcher import load_shards
from retrieval.arguments import parse_args
import logging
from retrieval.models import get_model
from retrieval.dataset import WikiChunkedDataset, EncodeCollator, EncodeDataset
from retrieval.searcher import WikiSearcher
from tensorflow.python.keras.testing_utils import use_gpu


def search_queries(retriever, q_reps, p_lookup, depth=10):
    # if args.batch_size > 0:
    #     all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    # else:
    all_scores, all_indices = retriever.search(q_reps, depth)
    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def tokenize_and_encode(prompt, model, device):
    tokenized_prompt = model.tokenize_text(prompt)
    tokenized_prompt = tokenized_prompt.to(device)
    q_reps = model.encode_query(tokenized_prompt)
    q_reps = q_reps.cpu().detach().numpy()
    return q_reps

def retrieve_text_from_indices(psg_indices, shard_map, data_args):

    psg_indices_flattened = psg_indices.flatten()
    shard_indices = np.array([shard_map[x] for x in psg_indices_flattened])

    # make a set of all the shard indices
    unique_shards = set(shard_indices.flatten())

    psg_text = {}
    for shard_idx in unique_shards:
        data_args.dataset_shard_index = shard_idx
        dset = WikiChunkedDataset(data_args=data_args)
        # get passage indices for this shard
        shard_psg_indices = np.where(shard_indices == shard_idx)
        # get the passage ids for this shard
        shard_psg_ids = psg_indices_flattened[shard_psg_indices]
        # get the passages for this shard

        shard_psg_text = [dset.get_idem_by_id(idx) for idx in shard_psg_ids]
        # store the passages
        for idx, psg in zip(shard_psg_ids, shard_psg_text):
            psg_text[idx] = psg
    # Map the passage text back to the passage indices psg_indices
    per_query_psg_text = []
    for q_idx in range(len(psg_indices)):
        q_psg_text = []
        for p_idx in psg_indices[q_idx]:
            q_psg_text.append(psg_text[p_idx])
        per_query_psg_text.append(q_psg_text)
    return per_query_psg_text


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():

    # model_args, data_args, training_args = parse_args()
    # # Load model
    # model = get_model(model_args, training_args)
    # model.to(training_args.device)
    #
    # index_files = glob.glob(data_args.encode_output_path)
    # print("Index files", index_files)
    #
    # logger = logging.getLogger(__name__)
    # logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')
    #
    # retriever, look_up, shard_map = load_shards(index_files)
    # print("Lookup length", len(look_up))
    # retriever.move_to_gpus()
    # prompt = ["What is the capital of China?", "What is the capital of Germany?"]
    # q_reps = tokenize_and_encode(prompt, model, training_args.device)
    # all_scores, all_indices =  search_queries(retriever, q_reps, look_up, depth=10)
    #
    # per_query_psg_text = retrieve_text_from_indices(all_indices, shard_map, data_args)
    #
    # print(per_query_psg_text[0], "\n\n", per_query_psg_text[1])

    searcher = WikiSearcher(document_reps = "/scratch/project/neural_ir/katya/IR_encoding/wiki_chunked/corpus/*",
                            # If you want to run it on one chunk only
                            # document_reps="/scratch/project/neural_ir/katya/IR_encoding/wiki_chunked/corpus/corpus.0.pkl"
                            dataset_cache_dir = "/scratch/project/neural_ir/katya/IR_datasets/wikipedia_2023_chunked/",
                            cache_dir = "/scratch/project/neural_ir/cache/hf_models/",
                            use_gpu = True)

    prompt = ["What is the capital of China?", "What is the capital of Germany?"]
    
    retrieved_text = searcher.search(prompt, k=2)
    print(retrieved_text)
    return True


if __name__ == '__main__':
    main()