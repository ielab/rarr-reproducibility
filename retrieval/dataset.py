import os.path
from typing import Tuple, List, Union
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer
# import nltk
from math import ceil

from retrieval.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)

class EncodeDataset(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=30
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.data = self.data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # text = self.data[item]
        return self.data[item]


class WikiDataset(EncodeDataset):
    def __getitem__(self, item) -> Tuple[str, str]:
        # Data sample contains the following fields: 'id', 'url', 'title', 'text'
        text = self.data[item]
        _format_passage = lambda title, txt: f"{title.strip()} {txt.strip()}".strip()
        formated_text = _format_passage(title=text['title'], txt=text['text'])
        text_id = text['id']
        return text_id, formated_text


class WikiChunkedDataset(Dataset):

    def __init__(self, dataset_cache_dir, dataset_shard_index):
        super().__init__()
        path_to_dset = os.path.join(dataset_cache_dir,
                                    "shards", f"shard_{dataset_shard_index}")
        self.data = load_from_disk(path_to_dset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item) -> Tuple[str, str]:
        # Data sample contains the following fields: 'id', 'url', 'title', 'text'
        text = self.data[item]
        _format_passage = lambda title, txt: f"{title.strip()} {txt.strip()}".strip()
        formated_text = _format_passage(title=text['title'], txt=text['text'])
        text_id = text['chunked_id']
        return text_id, formated_text

    def get_idem_by_id(self, id):
        list_of_ids = self.data['chunked_id']
        index = list_of_ids.index(id)
        return self.data[index]


@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len

        # tokenized_texts = []
        # # Tokenize and apply sliding window for each text
        # for text in texts:
        #     # Tokenize without truncation and special tokens, but we will add them manually
        #     tokens = self.tokenizer(
        #         text,
        #         padding=False,
        #         truncation=False,
        #         return_attention_mask=False,
        #         return_token_type_ids=False,
        #         add_special_tokens=True,
        #     ) ['input_ids']
        #
        #     # Apply sliding window if the text is longer than max_length
        #     if len(tokens) > max_length:
        #         logger.warning(f"Text is longer than max_length: {len(tokens)} > {max_length}")
        #     else:
        #         # If text is shorter than max_length, just append it
        #         tokenized_texts.append(tokens)
        tokens = self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        collated_texts = self.tokenizer.pad(
            tokens,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return text_ids, collated_texts