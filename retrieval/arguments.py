from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments
import logging


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    return model_args, data_args, training_args



@dataclass
class DataArguments:
    dataset_name: str = field(default="wikimedia/wikipedia", metadata={"help": "huggingface dataset name"})

    dataset_config: str = field(default="20231101.en",
                                metadata={"help": "huggingface dataset config, useful for datasets with sub-datasets"})

    dataset_path: str = field(default=None, metadata={"help": "Path to local data files or directory"})

    dataset_split: str = field(default='train', metadata={"help": "dataset split"})

    dataset_cache_dir: Optional[str] = field(default="/scratch/project/neural_ir/katya/IR_datasets/wikipedia_2023/",
                                             metadata={"help": "Where do you want to store the data downloaded from huggingface"})

    dataset_number_of_shards: int = field(default=1, metadata={"help": "number of shards to split the dataset into"})

    dataset_shard_index: int = field(default=0, metadata={"help": "shard index to use, to be used with dataset_number_of_shards"})

    chunked_dset_path: str = field(default="/scratch/project/neural_ir/katya/IR_datasets/wikipedia_2023_chunked/",
                                   metadata={"help": "Path to store chunked dataset"})

    encode_is_query: bool = field(default=False)

    encode_output_path: str = field(default=None, metadata={"help": "where to save the encode"})

    query_max_len: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    overlap_size: Optional[int] = field(
        default=32,
        metadata={"help": "The number of overlapping tokens between two windows."},
    )

    query_prefix: str = field(default='', metadata={"help": "prefix or instruction for query."})

    document_prefix: str = field(default='', metadata={"help": "prefix or instruction for document."})

    pad_to_multiple_of: Optional[int] = field(
        default=16,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to "
                    "enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})