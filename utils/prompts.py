import toml
import copy
import re
import json
import time


def create_llm_messages(prompt_config, prompt_data):
    """
    Renders a prompt template with data and creates messages for input to LLM.

    Input: (1) prompt_config to be used, (2) data used to render template
    Output: (1) structured messages for use as input to LLM
    """
    required_data = prompt_config["required_data"]["required"]
    missing_data = [data for data in required_data if data not in prompt_data]
    if missing_data:
        raise ValueError(f"Missing required data fields: {missing_data}")

    messages = copy.deepcopy(prompt_config["message_structure"]["messages"])
    # messages is a list of dictionaries
    for message in messages:
        # if content is dynamic
        if message["content"] in prompt_config["content"]["dynamic"]:
            # extract template and render content
            template = prompt_config["content"][message["content"]]
            message["content"] = template.format(**prompt_data)
        # otherwise retrieve from config content
        else:
            content_key = message["content"]
            message["content"] = prompt_config["content"][content_key]
    return messages


def load_prompt_config(method, model_family, use_modified_prompt=True):
    """
        Load the prompts from the toml file
        Modified prompt is revised from original RARR prompt to avoid malfunctions
    """

    if use_modified_prompt:
        path_to_config = f"./prompts/config_{method}_{model_family}_mod.toml"
    else:
        path_to_config = f"./prompts/config_{method}_{model_family}.toml"

    try:
        with open(path_to_config) as f:
            prompt_config = toml.load(f)
    except FileNotFoundError:
         print("File not found.")
         return None

    return prompt_config


def extract_json(text_response):
    # This pattern matches a string that starts with '{' and ends with '}'
    pattern = r'\{[^{}]*\}'
    matches = re.finditer(pattern, text_response)
    json_objects = []
    for match in matches:
        json_str = match.group(0)
        try:
            # Validate if the extracted string is valid JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Extend the search for nested structures
            extended_json_str = _json_extend_search(text_response, match.span())
            try:
                json_obj = json.loads(extended_json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                # Handle cases where the extraction is not valid JSON
                continue
    if json_objects:
        return json_objects
    else:
        return None  # Or handle this case as you prefer


def _json_extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            nest_count += 1
        elif text[i] == '}':
            nest_count -= 1
            if nest_count == 0:
                return text[start:i+1]
    return text[start:end]


def get_score_from_llm_response(llm_response):
    # First - drop all spaces, new lines and punctuation
    llm_response = re.sub(r'[^\w\s]|[\n]','',llm_response)

    match llm_response:
        case "Highly Relevant":
            return 2
        case "Somewhat Relevant":
            return 1
        case "Not Relevant":
            return 0
        case _:
            print(f"Unrecognized LLM response: \n{repr(llm_response)}")
            time.sleep(30)
            return -1
