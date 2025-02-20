from abc import ABC, abstractmethod
from typing import List

from openai import OpenAI

import torch
from torch import dtype

import os

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

class BaseLLM(ABC):
    def __init__(self,
                 model_name: str,
                 device: str | None = None):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def load_model(self):
        # loads model either from cloud or local machine - this method must be implemented by subclass
        pass

    @abstractmethod
    def generate(self, messages: List[dict]) -> str:
        # use the model to generate - this method must be implemented by subclass
        pass

class OpenAILLM(BaseLLM):
    def __init__(self,
                 model_name: str,
                 temperature: float,
                 device: str | None = None,
                 api_key_path: str | None = None):
        super().__init__(model_name, device)
        self.api_key_path = api_key_path
        self.api_key = self._load_key()
        self.temperature = temperature
        self.load_model()

    def _load_key(self):
        if self.api_key_path is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("No environment variable OPENAI_API_KEY")
            else:
                return api_key
        with open(self.api_key_path, 'r') as file:
            return file.read()

    def load_model(self):
        if self.api_key is None:
            raise ValueError("No API key provided")
        else:
            print("API key provided, ready to generate")

    def generate(self, messages: List[dict]) -> str:
        try:
            response = OpenAI(api_key=self.api_key)
            chat_params = {
                "messages": messages,
                "model": self.model_name,
                "response_format": {"type": "text"},
                "temperature": self.temperature,
                "max_tokens": 256
            }
            chat_completion = response.chat.completions.create(**chat_params)
            response_message = [choice.message.content.strip() for choice in chat_completion.choices]
            assert len(response_message) == 1
            return response_message[0]
        except Exception as e:
            print(f"Error: {e}")
            raise

class Llama3LLM(BaseLLM):
    def __init__(self,
                 model_name: str,
                 device: str | None = None,
                 temperature: float = 0.7,
                 torch_dtype: torch.dtype = torch.bfloat16,
                 model_cache_dir: str | None = None,
                 ):
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
        self.cache_dir = model_cache_dir
        self.tokenizer = None
        self.temperature = temperature
        self.model = None
        self.model_id = None
        self.set_model_id()
        self.load_model()


    def set_model_id(self):
        if self.model_name == "Llama-3.2-1B":
            self.model_id = "meta-llama/Llama-3.2-1B-Instruct"
        elif self.model_name == "Llama-3.2-3B":
            self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        elif self.model_name == "Llama-3.1-8B":
            self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif self.model_name == "Llama-3.1-70B":
            self.model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        else:
            raise ValueError(f"No matching model_id for model name: {self.model_name}")

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,
                                                       cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                                          # attn_implementation="flash_attention_2",
                                                          device_map="auto",
                                                          cache_dir=self.cache_dir,
                                                          torch_dtype=self.torch_dtype)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        set_seed(929)

    def generate(self, messages: List[dict], temp_increase: float = 0.0) -> List[str]:
        """messages is a list of dictionaries for a single prompt to the llm. Each dictionary has a key for  'role'
        and 'content'. """

        #TODO: implement transformers pipeline

        model = self.model

        # tokenize messages
        tokenized_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,)

        # convert tokenized_output to a tensor
        input_ids = torch.tensor([tokenized_input], dtype=torch.long)

        # create an attention mask where each non-pad token is 1 and each pad token is 0
        mask = (input_ids != self.tokenizer.pad_token_id)
        attention_mask = mask.to(dtype=torch.long)

        # since device map = auto was used we need to identify where the first layer is and send everything there
        main_device = next(model.parameters()).device
        input_ids = input_ids.to(main_device)
        attention_mask = attention_mask.to(main_device)

        # create input for model
        inputs = {'input_ids': input_ids, "attention_mask": attention_mask,}

        # print("generating...")
        generated_ids = model.generate(**inputs,
                                       max_new_tokens=512,
                                       do_sample=True,
                                       pad_token_id=self.tokenizer.eos_token_id,
                                       temperature=self.temperature + temp_increase)

        # print("decoding...")
        # decode newly generated tokens only (not tokens from prompt)
        generated_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response


def instantiate_llm(config):
    """
    Instantiates and returns an LLM object based on the given configuration.

    Args:
        config (dict): A dictionary containing keys such as:
            - "model_name" (str): The name of the model to load
              (e.g. "gpt-4", "Llama-3.2-1B", etc.).
            - "model_cache_dir" (str): Directory where the model is cached.
            - "temperature" (float): Sampling temperature for generation.

    Returns:
        llm: An instance of an LLM object (e.g., Llama3LLM or OpenAILLM).

    Raises:
        ValueError: If the specified model_name is not recognized.
    """
    model_name = config["model_name"]
    model_cache_dir = config["model_cache_dir"]

    llm = None

    # Check which model family is requested
    if model_name.startswith("Llama"):
        llm = Llama3LLM(
            model_name=model_name,
            model_cache_dir=model_cache_dir,
            temperature=config["temperature"]
        )
    elif model_name == "gpt-4":
        llm = OpenAILLM(
            model_name=model_name,
            temperature=config["temperature"]
        )

    # If no matching model was found, raise an error
    if llm is None:
        raise ValueError(f"LLM {model_name} not recognized")

    return llm




