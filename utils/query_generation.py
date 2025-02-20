"""Utils for running question generation."""
import os
import time
from idlelib.debugger_r import DictProxy
from typing import List, Dict
import jsonlines
import json
import tqdm
import sys
import openai

from utils import llms, prompts, query_generation


def parse_api_response(api_response: str) -> List[str]:
    """
    # TODO: update docstring
    Extract questions from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Question generation response from GPT-3.
    Returns:
        questions: A list of questions.
    """
    # print(f"api_response: {api_response}")
    search_string = "I googled:"
    questions = []
    for question in api_response.split("\n"):
        # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string)[1].strip()
        questions.append(question)

    return questions

def generate_queries(input_data, llm, config, fact_queries=None):
    """
    Generates a list of queries based on `config["query_strategy"]`:
      - "fact": directly return fact_queries
      - "rounds": use LLM calls for `config["num_rounds"]`
      - "baseline_fact": use length of fact queries
      - "baseline_count": use `config["baseline_query_count"]`
    """

    decon_sentence = input_data['decon_sentence']

    strategy = config["query_strategy"]

    if strategy == "fact":
        queries = input_data.get("fact_queries", [])
        return queries

    elif strategy == "rounds":
        if config["method"] == "query-gen-iterative":
            return _generate_queries_iterative(decon_sentence,llm, config)
        else:
            # Use your existing multi-round LLM code
            return _generate_queries_rounds(decon_sentence, llm, config)

    elif strategy == "baseline_fact":
        desired_count = len(input_data.get("fact_queries", []))
        return _generate_baseline_queries(decon_sentence, llm, config, desired_count)

    elif strategy == "baseline_count":
        baseline_count_key = config.get("baseline_count_key", -1)
        desired_count = input_data.get(baseline_count_key, -1)
        if (desired_count == -1) or (baseline_count_key == -1):
            raise ValueError(f"No baseline count key provided or baseline count not provided in input data")
        # adjust desired count by baseline offset - negative count not allowed
        desired_count = max(desired_count + config["baseline_count_offset"], 0)
        return _generate_baseline_queries(decon_sentence, llm, config, desired_count)

    else:
        raise ValueError(f"Unknown query_strategy: {strategy}")

def _load_prompt_and_messages(decon_sentence, config):
    model_family = "Llama"
    prompt_config = prompts.load_prompt_config(
        method=config['method'],
        model_family=model_family,
        use_modified_prompt=config["use_modified_prompt"]
    )
    return prompts.create_llm_messages(
        prompt_config=prompt_config,
        prompt_data={"claim": decon_sentence}
    )
def _load_prompt_and_messages_iterative(decon_sentence, existing_queries, config):
    model_family = "Llama"
    prompt_config = prompts.load_prompt_config(
        method=config['method'],
        model_family=model_family,
        use_modified_prompt=config["use_modified_prompt"]
    )
    return prompts.create_llm_messages(
        prompt_config=prompt_config,
        prompt_data={"claim": decon_sentence, "existing_queries": existing_queries}
    )

def _generate_queries_rounds(decon_sentence, llm, config):
    """
    Generates a set of queries from a given sentence using an LLM.

    Args:
        decon_sentence (str): The decontextualized sentence you want to create queries about.
        llm (Any): The instantiated LLM object (from your `llms.instantiate_llm`).
        config (Dict): A configuration dictionary that must include:
            - "num_rounds": (int) Number of times to query the LLM.
            - "use_modified_prompt": (bool) Whether to use a "modified" prompt.
            - "model_family": (str) Model family (e.g., "Llama", "GPT-3").

    Returns:
        List[str]: A list of generated queries (unique, deduplicated, and sorted).
    """

    messages = _load_prompt_and_messages(decon_sentence, config)

    num_rounds = config["num_rounds"]

    # Run the LLM multiple rounds to gather different questions
    queries = set()
    llm_responses = []
    for _ in range(num_rounds):
        try:
            response_message = llm.generate(messages=messages)
            llm_responses.append(response_message)

            # Extract questions from response
            cur_round_queries = query_generation.parse_api_response(response_message.strip())
            queries.update(cur_round_queries)

        except Exception as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)

    # Sort and convert set to list
    queries_list = sorted(list(queries))

    return queries_list

def _generate_queries_iterative(decon_sentence, llm, config):
    """
    Generates a set of queries from a given sentence using an LLM.

    Args:
        decon_sentence (str): The decontextualized sentence you want to create queries about.
        llm (Any): The instantiated LLM object (from your `llms.instantiate_llm`).
        config (Dict): A configuration dictionary that must include:
            - "num_rounds": (int) Number of times to query the LLM.
            - "use_modified_prompt": (bool) Whether to use a "modified" prompt.
            - "model_family": (str) Model family (e.g., "Llama", "GPT-3").

    Returns:
        List[str]: A list of generated queries (unique, deduplicated, and sorted).
    """
    num_rounds = config["num_rounds"]

    # Run the LLM multiple rounds to gather different questions
    queries = set()
    llm_responses = []
    for i in range(num_rounds):
        try:
            if queries:
                # Possibly sort them to keep stable order
                existing_queries_list = sorted(queries)
                existing_queries_str = "\n".join(
                    f"{index+1}. {q}"
                    for index, q in enumerate(existing_queries_list)
                )
            else:
                existing_queries_str = ""

            messages = _load_prompt_and_messages_iterative(decon_sentence, existing_queries_str, config)
            response_message = llm.generate(messages=messages)
            llm_responses.append(response_message)

            # Extract questions from response
            cur_round_queries = query_generation.parse_api_response(response_message.strip())
            queries.update(cur_round_queries)

        except Exception as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)

    # Sort and convert set to list
    queries_list = sorted(list(queries))

    return queries_list

def _generate_baseline_queries(decon_sentence, llm, config, desired_count):
    """
    Generates a (pseudo) 'ordered set' of queries from a given sentence using an LLM.
    Returns them in insertion order (first-come, first-served), up to `desired_count`.
    """

    messages = _load_prompt_and_messages(decon_sentence, config)
    max_attempts = config["max_attempts"]

    # use a dict instead of a set to preserve insertion order
    queries_dict = {}
    llm_responses = []
    counter = 0
    temp_increase = 0

    # keep going until we have at least `desired_count` unique queries
    while len(queries_dict) < desired_count:
        try:
            response_message = llm.generate(messages=messages, temp_increase=temp_increase)
            llm_responses.append(response_message)

            # Extract questions from response
            cur_round_queries = query_generation.parse_api_response(response_message.strip())
            # Insert them into the dict (preserves the order in which we encounter them)
            for q in cur_round_queries:
                # Only insert if it's not already in our dict
                if q not in queries_dict:
                    queries_dict[q] = None

        except Exception as exception:
            print(f"{exception}. Retrying...")
            time.sleep(1)

        counter += 1

        # If we haven't reached the desired_count yet, but have hit max_attempts,
        # bump temperature to try to get more variety
        if counter >= max_attempts and len(queries_dict) < desired_count:
            print(f"Unable to generate {desired_count} unique queries after {max_attempts} attempts., "
                  f"raising temperature by 0.05")
            print(f"sentence: {decon_sentence}")
            print(f"queries:{list(queries_dict.keys())}")

            temp_increase += 0.05

        if counter >= 3 * max_attempts:
            raise RuntimeError(
                f"Unable to generate {desired_count} unique queries after {3 * max_attempts} attempts."
            )

    # Now take the first `desired_count` in the order they were inserted
    queries_list = list(queries_dict.keys())[:desired_count]

    return queries_list




def run_query_gen(config):
    """
    Reads from an input file (JSON Lines format), generates queries for each item,
    and writes the results to an output file.

    If `use_fact_queries` is True, queries come directly from `item["fact_queries"]`
    and no LLM call is made. Otherwise, queries are generated by the LLM.

    Args:
        config (Dict): A configuration dictionary containing:
            - "use_fact_queries": bool
            - "input_file": str
            - "output_file": str
            - "num_rounds": int
            - "use_modified_prompt": bool
            - "model_family": str (optional)
            - Additional keys as needed by your code
    """
    # instantiate llm
    llm = llms.instantiate_llm(config)

    # iterate over the input sentences
    with jsonlines.open(config["input_file"]) as input_reader, \
            open(config["output_file"], "w", encoding="utf-8") as output_writer:

        for i, item in enumerate(tqdm.tqdm(input_reader), start=1):
            # start timing the iteration
            start_time = time.perf_counter()

            output_data = item

            # TODO: make model family dynamic if using other model families
            queries = generate_queries(input_data=output_data, llm=llm, config=config)

            iteration_time = time.perf_counter() - start_time
            output_data["qg_iteration_time"] = iteration_time

            # # print iteration info to stdout
            # print(f" Iteration {i} query gen completed in {iteration_time:.4f} seconds", file=sys.stderr, end="")

            # add queries to output data
            output_data["queries"] = queries


            # write 1 line of output
            output_writer.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            output_writer.flush()



def run_rarr_question_generation(
    claim: str,
    llm: llms.BaseLLM,
    model: str,
    prompt: str,
    num_rounds: int,
    original_query: str=None,
    context: str = None,
    num_retries: int = 5,
    use_modified_prompts: bool = False,
) -> Dict[str, List[str] | str]:
    """
    #TODO: update docstring
    Generates questions that interrogate the information in a claim.

    Given a piece of text (claim), we use GPT-3 to generate questions that question the
    information in the claim. We run num_rounds of sampling to get a diverse set of questions.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_rounds: Number of times to sample questions.
    Returns:
        questions: A list of questions.
    """
    # TODO: context not implemented
    if context:
        gpt3_input = prompt.format(context=context, claim=claim).strip()
    else:
        # create messages from prompt configuration file
        # load prompt config
        # print(f"claim: {claim}")
        if use_modified_prompts:
            print("using modified qgen prompt")
            if original_query is not None:

                # Delete "Question:" from the original query
                original_query = original_query.replace("Question: ", "").strip()

                prompt_config = load_prompt_config(component_label='question-gen-modv2', model_name=model)
                messages = create_llm_messages(prompt_config=prompt_config, prompt_data={"claim": claim,
                                                                                         "query": original_query})
               #  print(f"Messages for QGEN: {messages}")
            else:
                prompt_config = load_prompt_config(component_label='question-gen-mod', model_name=model)
                messages = create_llm_messages(prompt_config=prompt_config, prompt_data={"claim": claim})


        else:
            prompt_config = load_prompt_config(component_label='question-gen', model_name=model)
            messages = create_llm_messages(prompt_config=prompt_config, prompt_data={"claim": claim})

    questions = set()
    responses = []
    for _ in range(num_rounds):
        for _ in range(num_retries):
            try:
                # print(f"messages: {messages}")
                response_message = llm.generate(messages=messages)
                responses.append(response_message)

                # extract questions from response
                cur_round_questions = parse_api_response(response_message.strip())

                questions.update(cur_round_questions)
                break
            except Exception as exception:
                print(f"{exception}. Retrying...")
                time.sleep(1)

    questions = list(sorted(questions))

    return {"questions": questions, "responses": responses}
