import json
import textwrap
import evaluate
from datasets import Dataset
from colorama import Fore, Style
import os 

# from stream_bench.llms.oai_chat import OpenAIChat
from abc import ABC, abstractmethod
from typing import Any
from datasets import load_dataset, Dataset

import time
import groq
import openai
import together
import anthropic
import google.api_core.exceptions as g_exceptions
import urllib.request
from colorama import Fore, Style


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 0.25,
    exponential_base: float = 2,
    max_retries: int = 10
):
    # Define errors based on available libraries.
    errors_tuple = (
        openai.RateLimitError, openai.APIError,
        g_exceptions.ResourceExhausted, g_exceptions.ServiceUnavailable, g_exceptions.GoogleAPIError,
        anthropic.BadRequestError, anthropic.InternalServerError, anthropic.RateLimitError,
        urllib.error.HTTPError, urllib.error.URLError,
        groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError,
        together.error.TogetherException,
        ValueError, IndexError, UnboundLocalError
    )
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors_tuple as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if isinstance(e, ValueError) or (num_retries > max_retries):
                    print(Fore.RED + f"ValueError / Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL)
                    result = 'error:{}'.format(e)
                    prompt = kwargs["prompt"] if "prompt" in kwargs else args[1]
                    res_info = {
                        "input": prompt,
                        "output": result,
                        "num_input_tokens": len(prompt) // 4,  # approximation
                        "num_output_tokens": 0,
                        "logprobs": []
                    }
                    return result, res_info
                # Sleep for the delay
                print(Fore.YELLOW + f"Error encountered ({e}). Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                time.sleep(delay)
                # Increment the delay
                delay *= exponential_base
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper


def extract_json_string(res: str) -> str:
    """Extract the first valid json string from the response string (of LLMs).
    
    Return '' (empty string) if not found. Raise ValueError if an } is found before any {.
    """
    start, end = -1, -1
    cnt = 0  # act as a representation of a stack of '{' '}' pairs
    for i in range(len(res)):
        ch = res[i]
        if ch == '{':
            if cnt == 0:  # the first time '{' is encountered
                start = i
            cnt += 1
        elif ch == '}':
            if cnt <= 0:
                raise ValueError("found } before any { appears")
            cnt -= 1
            if cnt == 0:  # found the end index
                end = i
                break
    return res[start:end+1]


def strip_all_lines(s: str) -> str:
    """Remove all leading and trailing spaces of each line in the string."""
    return '\n'.join([line.strip() for line in s.splitlines()])


class LLM(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """Setup LLM configs here"""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.0, **kwargs) -> tuple[str, dict]:
        """Prompt the LLM and get a tuple of (response_text, response_info).
        
        The response_info should be in the following format:
        {
            "input": prompt,
            "output": <response_text>,
            "num_input_tokens": <number_of_input_tokens>,
            "num_output_tokens": <number_of_output_tokens>,
            "logprobs": <log_probs_of_each_token_position>  # please refer to oai_chat.py for the schema
        }
        """
        raise NotImplementedError
        

class OpenAIChat(LLM):
    TOP_LOGPROBS = 1

    def __init__(self, model_name='gpt-3.5-turbo-0125') -> None:
        params = {'api_key': os.environ['OAI_KEY'], 'organization': os.environ['OAI_ORG']}
        if os.getenv('CUSTOM_API_URL'):
            params['base_url'] = os.environ['CUSTOM_API_URL']
        self.client = openai.OpenAI(**params)
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.0, **kwargs) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            logprobs=True,
            top_logprobs=self.TOP_LOGPROBS,
            **kwargs
        )
        log_prob_seq = response.choices[0].logprobs.content
        assert response.usage.completion_tokens == len(log_prob_seq)
        res_text = response.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
            "logprobs": [[{"token": pos_info.token, "logprob": pos_info.logprob} for pos_info in position.top_logprobs] for position in log_prob_seq]
        }
        return res_text, res_info


class Bench(ABC):
    """Associated with corresponding Dataset, Feedback, and Metrics"""
    DATASET_PATH: str = None
    DATASET_NAME: str = None

    def __init__(self, config: dict):
        self.config = config
        self.dataset = load_dataset(self.DATASET_PATH, self.DATASET_NAME)
        self.use_wandb = False
        self.n_correct = 0
        self.references = []
        self.predictions = []

    @abstractmethod
    def get_dataset(self) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def give_feedback(self, model_output: str, row: dict, res: dict) -> tuple[bool, dict]:
        """Give feedback according to benchmark configurations.

        1. No feedback
        2. Weak feedback (True: correct, False: incorrect)
        3. Strong feedback (Ground truth string)
        """
        raise NotImplementedError

    @abstractmethod
    def get_input(self, row: dict) -> dict:
        """
        return must include prompt field of string value
        """
        raise NotImplementedError

    @abstractmethod
    def get_output(self, row: dict) -> dict:
        """Extract the label of the row (instance)."""
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self) -> dict:
        """Calculate and return the metrics for this benchmark."""
        raise NotImplementedError

    @abstractmethod
    def postprocess_generation(self, res: Any, idx: int) -> Any:
        """Postprocess the agent's response and map it to the label space.

        Args:
            res (Any): the agent's response
            idx (int): the instance's index in the dataset

        Returns:
            Any: the predicted label
        """
        raise NotImplementedError

    @abstractmethod
    def process_results(self, prediction: Any, label: Any, return_details: bool = False) -> bool | dict:
        """Compare the prediction with the label and calculate streaming metrics at the current time point.

        Args:
            prediction (Any): the agent's prediction
            label (Any): the ground truth label
            return_details (bool, optional): Return correctness of prediction (bool) or detailed metrics (dict). Defaults to False

        Returns:
            bool | dict:  Return correctness of prediction (bool) or detailed metrics (dict)
        """
        raise NotImplementedError
        

class GSM8KBench(Bench):
    """Benchmark for the GSM8K dataset."""
    DATASET_PATH = "appier-ai-research/robust-finetuning"
    DATASET_NAME = "gsm8k"
    EVAL_LLM = "gpt-4o-mini-2024-07-18"  # for extracting the answer from LLMs' raw output

    def __init__(
        self,
        split: str = "test",
        seed: int = 42,
        feedback: str = "correctness",
        **kwargs
    ) -> None:
        super().__init__({})
        self.split = split
        self.seed = seed
        self.feedback = feedback
        self.eval_func = evaluate.load("exact_match")
        self.llm = OpenAIChat(model_name=self.EVAL_LLM)

    def get_dataset(self) -> Dataset:
        return self.dataset[self.split].shuffle(seed=self.seed)

    @staticmethod
    def get_zeroshot_prompt(question: str) -> str:
        return question

    def get_input(self, row: dict) -> dict:
        row_input = dict()
        row_input["question"] = row["question"]
        row_input["prompt_zeroshot"] = self.get_zeroshot_prompt(row["question"])
        return row_input

    def get_output(self, row: dict) -> dict:
        rna = row["answer"].split("####")
        rationale = rna[0].strip()
        answer = rna[1].strip()
        if ',' in answer:
            answer = ''.join(answer.split(','))
        return {"rationale": rationale, "answer": answer}

    def get_metrics(self) -> dict:
        metrics = self.eval_func.compute(
            predictions=self.predictions,
            references=self.references,
            ignore_punctuation=True
        )
        return metrics

    def postprocess_generation(self, res: str, idx: int = -1) -> str:
        # Parse out the answer with a cheap LLM
        prompt_extract = strip_all_lines("""\
        The following text is an LLM's response to a math question:

        Text (enclosed in triple quotes): '''{text}'''

        Extract the answer from the text (only extract the digits, potentially the sign if the number is negative), and provide it in the following JSON format:
        {{"answer": "<digits>"}}""")
        prompt = prompt_extract.format(text=res)
        answer_str, _ = self.llm(prompt)
        answer_json = extract_json_string(answer_str)
        try:
            answer = json.loads(answer_json)["answer"]
        except (json.JSONDecodeError, KeyError) as e:
            print(Fore.RED + str(e) + Style.RESET_ALL)
            answer = res
        return answer

    def process_results(
        self,
        prediction: str,
        label: dict,
        return_details: bool = True,
        **kwargs
    ) -> bool | dict:
        answer = label["answer"]
        correct = self.eval_func.compute(
            predictions=[prediction],
            references=[answer]
        )["exact_match"]
        self.n_correct += correct
        self.predictions.append(prediction)
        self.references.append(answer)
        if return_details:
            return {
                "correct": int(correct),
                "n_correct": self.n_correct,
                "rolling_em": self.get_metrics()["exact_match"]
            }
        return bool(correct)

    def give_feedback(self, model_output: str, row: dict, res: dict) -> tuple[bool, dict]:
        return True, {}