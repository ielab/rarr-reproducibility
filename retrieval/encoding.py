
import os
import pickle
from tqdm import tqdm
import numpy as np
from contextlib import nullcontext

from typing import List
# import nltk

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import pyarrow as pa

from retrieval.models import get_model
from retrieval.dataset import WikiChunkedDataset, EncodeCollator, EncodeDataset
from retrieval.arguments import parse_args


def encode(model, encode_loader, training_args, data_args):
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.amp.autocast('cuda') if training_args.fp16 or training_args.bf16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_query:
                    q_reps = model.encode_query(batch)
                    encoded.append(q_reps.cpu().detach().numpy())
                else:
                    p_reps = model.encode_document(batch)
                    encoded.append(p_reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded)

    return encoded, lookup_indices



def main():
    model_args, data_args, training_args = parse_args()

    # Load model
    model = get_model(model_args, training_args)

    # Setup dataset
    
    encode_dataset = WikiChunkedDataset(data_args.dataset_cache_dir, data_args.dataset_shard_index)
    encode_collator = EncodeCollator(data_args=data_args, tokenizer=model.tokenizer)
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    # Encode the dataset
    encoded, lookup_indices = encode(model, encode_loader, training_args, data_args)

    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

    return True


if __name__ == "__main__":
    main()
