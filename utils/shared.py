import yaml
import argparse
import os
import sys

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_args() -> argparse.Namespace:
    """Gets command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        required=True,
        help="path to experiment config file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["wiki", "fava"],
        help="name of dataset",
    )
    args = parser.parse_args()

    return args


def get_start_line(config):
    """
    Determine the line from which to resume based on whether we are resuming
    and whether the output file already exists.

    :param config: A dictionary-like config that must include 'output_file' key.
    :return: The zero-based line index from which to resume.
    """
    output_file = config.get("output_file", None)
    if output_file and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as out_f:
            start_line = sum(1 for _ in out_f)
        print(f"Resuming from line: {start_line}", file=sys.stderr)
    else:
        start_line = 0

    return start_line