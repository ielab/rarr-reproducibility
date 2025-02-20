"""
input: queries for each sentence from query gen
output: evidence for the queries for each sentence


"""

import json, jsonlines
import tqdm
import datetime
import yaml
import os
import time
import argparse
import sys

from retrieval.searcher import WikiSearcher

from utils.shared import get_start_line

def run_evidence_retrieval(config):

    model_cache_dir = config['model_cache_dir']

    # instantiate wikipedia corpus searcher
    # TODO: stop downloading new stella snapshots(see retrieval.models.StellaEncoder   )
    searcher = WikiSearcher(document_reps_dir=config["document_reps_dir"],
                            retrieval_model_name=config["retrieval_model_name"],
                            dataset_cache_dir=config["corpus_cache_dir"],
                            model_cache_dir=model_cache_dir,
                            use_gpu=True,
                            debug=config["debug"])

    # check if we should resume from a previous run
    resume = config.get("resume", False)

    if resume:
        start_line = get_start_line(config)
        print(f"Resuming evidence retrieval from line {start_line}")
    else:
        start_line = 0

    # ff resuming, open output in append mode; else overwrite
    write_mode = 'a' if resume else 'w'

    # iterate over the input sentences
    with jsonlines.open(config["input_file"]) as input_reader, \
            open(config["output_file"], write_mode, encoding="utf-8") as output_writer:

        for i, item in enumerate(input_reader):
            # if we haven't reached start line
            if i < start_line:
                continue

            # start timing the iteration
            start_time = time.perf_counter()

            output_data = item

            # extract queries
            queries = item.get("queries", [])

            if not queries:
                # handle no queries
                evidence = []
                relevance_matrix = []
            else:
                evidence, relevance_matrix = searcher.retrieve_relevant_passages(
                    queries,
                    k=config["max_evidence_per_query"]
                )

            # add evidence to queries
            output_data["retrieved_evidence"] = evidence

            iteration_time = time.perf_counter() - start_time
            output_data["ret_iteration_time"] = iteration_time

            # print iteration info to stdout
            print(f"Iteration {i} retrieval completed in {iteration_time:.4f} seconds", file=sys.stderr, flush=True)

            # write 1 line of output
            output_writer.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            output_writer.flush()
