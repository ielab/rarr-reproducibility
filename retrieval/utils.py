import pickle
import numpy as np


def read_trec_run(file):
    run = {}
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in run:
                run[qid] = {'docs': {}, 'max_score': float(score), 'min_score': float(score)}
            run[qid]['docs'][docid] = float(score)
            run[qid]['min_score'] = float(score)
    return run


def read_query_file(query_file_path):
    queries = {}
    with open(query_file_path, 'r') as f:
        for line in f:
            qid, query = line.strip().split('\t')
            queries[qid] = query
    return queries


def write_trec_run(run, file, name='hybrid'):
    with open(file, 'w') as f:
        for qid in run:
            doc_score = run[qid]
            if 'docs' in doc_score:
                doc_score = doc_score['docs']
            # sort by score
            doc_score = dict(sorted(doc_score.items(), key=lambda item: item[1], reverse=True))
            for i, (doc, score) in enumerate(doc_score.items()):
                f.write(f'{qid} Q0 {doc} {i+1} {score} {name}\n')


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file, max_p, max_p_delimiter, depth=None):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            unique_docs = set()
            rank = 1
            for s, idx in score_list:
                if max_p:
                    idx = idx.split(max_p_delimiter)[0]
                    if idx in unique_docs:
                        continue
                    unique_docs.add(idx)

                # trec run formate
                f.write(f'{qid} Q0 {idx} {rank} {s} ielab\n')
                rank += 1
                if depth is not None and rank > depth:
                    break


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup