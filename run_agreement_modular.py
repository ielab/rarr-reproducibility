"""
Runs RARR agreement using modular components
"""

import json, jsonlines
import tqdm
import time
import datetime
import os
import yaml
import argparse
import sys

from utils.query_generation import run_query_gen
from utils.evidence_retrieval import run_evidence_retrieval
from utils.agreement_gate import run_agreement

from utils.shared import load_config, get_args

def load_yaml_with_variable(file_path, var_name, var_value):
    """Load a YAML file, replacing any instances of '${var_name}' with var_value."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace the placeholder with var_value
    placeholder = f"${{{var_name}}}"  # e.g. ${DATASET}
    content = content.replace(placeholder, var_value)

    # Now parse the replaced string as YAML
    return yaml.safe_load(content)

def main() -> None:

    args = get_args()

    # Load the configuration
    config_all = load_yaml_with_variable(file_path=args.config_file_path, var_name="DATASET", var_value=args.dataset_name)

    config_qg = config_all['query_gen']
    config_ret = config_all['evidence_retrieval']
    config_agr = config_all['agreement']

    #############
    # query gen #
    #############

    # check for existing data
    if os.path.exists(config_qg["output_file"]):
        print(f"{config_qg['output_file']} already exists, skipping query generation.", file=sys.stderr)
    else:
        print(f"Running query generation for: {config_qg['output_file']}", file=sys.stderr)
        run_query_gen(config_qg)

    #############
    # retrieval #
    #############

    # check for existing data and resume flag not set
    if os.path.exists(config_ret["output_file"]) and not config_ret['resume']:
        print(f"{config_ret['output_file']} already exists, skipping evidence retrieval.", file=sys.stderr)
    else:
        print(f"Running evidence retrieval for: {config_ret['output_file']}", file=sys.stderr)
        run_evidence_retrieval(config_ret)

    #############
    # agreement #
    #############

    # check for existing data
    if os.path.exists(config_agr["output_file"]) and not config_agr['resume']:
        print(f"{config_agr['output_file']} already exists, skipping agreement.", file=sys.stderr)
    else:
        print(f"Running agreement for: {config_agr['output_file']}", file=sys.stderr)
        run_agreement(config_agr)


    # save configuration
    timestamp = datetime.datetime.now().strftime('%d_%m_%y_%H%M')
    experiment_basename = os.path.splitext(config_agr["output_file"])[0]
    config_save_name = f"{experiment_basename}_config_{timestamp}.yaml"

    with open(config_save_name, "w") as file:
        yaml.safe_dump(config_all, file)

    print(f"Configuration saved as: {config_save_name}")


if __name__ == "__main__":
    main()
