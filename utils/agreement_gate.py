"""Utils for running the agreement gate."""
import os
import time
import json
from typing import Any, Dict, Tuple
import jsonlines
import tqdm
import openai
import sys

from utils import llms, prompts, agreement_gate

from utils.shared import get_start_line

openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_api_response(api_response: str) -> Tuple[bool, str, str, int]:
    """
    # TODO: update docstring
    Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Agreement gate response from GPT-3.
    Returns:
        is_open: Whether the agreement gate is open.
        reason: The reasoning for why the agreement gate is open or closed.
        decision: The decision of the status of the gate in string form.
    """
    api_response = api_response.strip().split("\n")
    if len(api_response) < 2:
        reason = "Failed to parse."
        decision = None
        is_open = False
        print(f"Failed to parse response.: {api_response}")
        score = -3
        # time.sleep(20)
    else:
        reason = api_response[0]
        decision = api_response[1].split("Conclusion:")[-1].strip()

        if 'disagrees' in decision:
            score = 1
        elif 'agrees' in decision:
            score = 0
        elif 'irrelevant' in decision:
            score = 2
        elif decision == '':
            print(f"==== Decision is empty: ==== \n {api_response}")
            score = -1
        else:
            print(f"==== Unknown decision: ==== \n  {api_response}")
            score = -2
            # pause for 30 seconds
        is_open = "disagrees" in api_response[1]

    return is_open, reason, decision, score


def run_agreement(config):
    # check if we should resume from a previous run
    resume = config.get("resume", False)

    if resume:
        start_line = get_start_line(config)
        print(f"Resuming agreement checks from line {start_line}")
    else:
        start_line = 0

    # ff resuming, open output in append mode; else overwrite
    write_mode = 'a' if resume else 'w'

    llm = llms.instantiate_llm(config)
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

            # extract decontextualized sentence
            decon_sentence = item["decon_sentence"]

            # load the correct prompt configuration
            #TODO: make model family dynamic if using other model families
            prompt_config = prompts.load_prompt_config(method='agreement',
                                                       model_family="Llama",
                                                       use_modified_prompt=config["use_modified_prompt"])

            # TODO: add check to ensure we are getting the number of queries and evidence we expect

            # track agreement gates for each sentence
            agreement_gates = []

            # iterate over queries
            for idx_q, query in enumerate(output_data["queries"]):

                # iterate over each query's evidence
                for idx_e, evidence in enumerate(output_data["retrieved_evidence"][idx_q]):

                    # create llm messages
                    messages = prompts.create_llm_messages(prompt_config=prompt_config,
                                                           prompt_data={"claim": decon_sentence,
                                                                        "query": query,
                                                                        "evidence": evidence["text"]})

                    response_message = llm.generate(messages=messages)

                    is_open, reason, decision, score = agreement_gate.parse_api_response(response_message)

                    gate = {"is_open": is_open, "reason": reason, "decision": decision, "score": score}

                    gate["query"] = query
                    gate["decon_sentence"] = decon_sentence
                    gate["evidence"] = evidence["text"]

                    agreement_gates.append(gate)

            # add agreement gate to output data
            output_data["agreement_gates"]  = agreement_gates

            iteration_time = time.perf_counter() - start_time
            output_data["agr_iteration_time"] = iteration_time

            # print iteration info to stdout
            print(f"Iteration {i} agreement completed in {iteration_time:.4f} seconds", file=sys.stderr, flush=True)

            # write 1 line of output
            output_writer.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            output_writer.flush()




def run_agreement_gate(
    claim: str,
    llm: llms.BaseLLM,
    query: str,
    evidence: str,
    model: str,
    prompt: str,
    context: str = None,
    num_retries: int = 5,
    use_modified_prompts: bool = False,
) -> Dict[str, Any]:
    """
    #TODO: update docstring

    Checks if a provided evidence contradicts the claim given a query.

    Checks if the answer to a query using the claim contradicts the answer using the
    evidence. If so, we open the agreement gate, which means that we allow the editor
    to edit the claim. Otherwise the agreement gate is closed.

    Args:
        claim: Text to check the validity of.
        query: Query to guide the validity check.
        evidence: Evidence to judge the validity of the claim against.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        gate: A dictionary with the status of the gate and reasoning for decision.
    """
    # TODO: context not implemented
    if context:
        gpt3_input = prompt.format(
            context=context, claim=claim, query=query, evidence=evidence
        ).strip()
    # load prompt config and create messages from prompt configuration file
    else:
        # print(f"claim: {claim}")
        if use_modified_prompts:
            print("using modified agreement prompt")
            prompt_config = load_prompt_config(component_label='agreement-mod', model_name=model)
        else:
            prompt_config = load_prompt_config(component_label='agreement', model_name=model)
        messages = create_llm_messages(prompt_config=prompt_config, prompt_data={"claim": claim,
                                                                                 "query": query,
                                                                                 "evidence": evidence})

        # print(f"messages: {messages}")

    for _ in range(num_retries):
        try:
            response_message = llm.generate(messages=messages)
            break
        except Exception as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)

    is_open, reason, decision, score = parse_api_response(response_message)
    gate = {"is_open": is_open, "reason": reason, "decision": decision, "score": score}
    return gate
