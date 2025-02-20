"""Utils for running the editor."""
import os
import time
# import json_repair
from typing import Dict, Union
import json
import re

import openai

from .llms import BaseLLM
from .prompts import create_llm_messages, load_prompt_config

openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_api_response(api_response: str, use_modified_prompts: bool) -> str:
    """
    # TODO: update docstring
    Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns a reason for the edit and the edit in two consecutive lines.
    Only extract out the edit from the second line.

    Args:
        api_response: Editor response from GPT-3.
    Returns:
        edited_claim: The edited claim.
    """
    # api_response = api_response.strip().split("\n")
    # if len(api_response) < 2:
    #     print("Editor error.")
    #     return None
    # edited_claim = api_response[1].split("My fix:")[-1].strip()
    # return edited_claim

    api_response = api_response.strip()

    # revised prompt
    if use_modified_prompts:
        edited_claim = api_response.split("Revised claim:")[-1].strip()
    # original rarr prompt
    else:
        edited_claim = api_response.split("My fix:")[-1].strip()

    return edited_claim


def run_rarr_editor(
    claim: str,
    llm: BaseLLM,
    query: str,
    evidence: str,
    model: str,
    prompt: str,
    context: str = None,
    num_retries: int = 5,
    use_modified_prompts: bool = False,
) -> Dict[str, str | bool]:
    """
    #TODO: update docstring
    Runs a GPT-3 editor on the claim given a query and evidence to support the edit.

    Args:
        claim: Text to edit.
        query: Query to guide the editing.
        evidence: Evidence to base the edit on.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        edited_claim: The edited claim.
    """
    #TODO: context not implemented
    if context:
        gpt3_input = prompt.format(
            context=context, claim=claim, query=query, evidence=evidence
        ).strip()
    else:
        print(f"claim: {claim}")
        if use_modified_prompts:
            print("using modified editor prompt")
            prompt_config = load_prompt_config(component_label='edit-mod', model_name=model)
        else:
            # load prompt config and create messages from prompt configuration file
            prompt_config = load_prompt_config(component_label='edit', model_name=model)
        messages = create_llm_messages(prompt_config=prompt_config, prompt_data={"claim": claim,
                                                                                 "query": query,
                                                                                 "evidence": evidence})
    for _ in range(num_retries):
        try:
            response_message = llm.generate(messages=messages)
            break
        except Exception as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)

    edited_claim = parse_api_response(response_message, use_modified_prompts=use_modified_prompts)


    output = {"edited_claim": edited_claim, "malfunction": False}
    # If there was an error in GPT-3 generation, return the claim.
    if not edited_claim:
        output["edited_claim"] = claim
        output["malfunction"] = True

    return output
