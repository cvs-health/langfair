# Copyright 2024 CVS Health and/or one of its affiliates
#
# Copyright 2023 OpenAI
#
# Licensed under the MIT License.
#
# The original work of OpenAI has been modified
# by CVS Health to include functionality for computing
# prompt and response token counts for OpenAI models.

import asyncio
import itertools
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import langchain_core
import numpy as np
import tiktoken
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

from langfair.constants.cost_data import COST_MAPPING, FAILURE_MESSAGE, TOKEN_COST_DATE

N_PARAM_WARNING = """
The 'use_n_param' parameter may not be compatible with all BaseChatModel instances. 
Please ensure that your specific BaseChatModel has an 'n' attribute and supports setting 'n' to a value up to 'count'.
Note that some BaseChatModel instances only support 'n' up to a certain value. If 'count' exceeds this value, an error may occur.
"""


class ResponseGenerator:
    def __init__(
        self,
        langchain_llm: Any = None,
        suppressed_exceptions: Optional[
            Union[Tuple[BaseException], BaseException, Dict[BaseException, str]]
        ] = None,
        use_n_param: bool = False,
        max_calls_per_min: Optional[int] = None,
    ) -> None:
        """
        Class for generating data from a provided set of prompts

        Parameters
        ----------
        langchain_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `langchain_llm` object.

        suppressed_exceptions : tuple or dict, default=None
            If a tuple, specifies which exceptions to handle as 'Unable to get response' rather than raising the
            exception. If a dict, enables users to specify exception-specific failure messages with keys being subclasses
            of BaseException

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when count > 1.

        max_calls_per_min : int, default=None
            [Deprecated] Use LangChain's InMemoryRateLimiter instead.
        """
        self.cost_mapping = COST_MAPPING
        self.token_cost_date = TOKEN_COST_DATE
        self.llm = langchain_llm
        self.use_n_param = use_n_param
        if isinstance(suppressed_exceptions, Dict):
            if self._valid_exceptions(tuple(suppressed_exceptions.keys())):
                self.suppressed_exceptions = suppressed_exceptions
        elif self._valid_exceptions(suppressed_exceptions):
            self.suppressed_exceptions = suppressed_exceptions
        else:
            raise TypeError(
                """suppressed_exceptions must be a subclass of BaseException or a tuple of subclasses of BaseException 
                or a Dict with keys being subclasses of BaseException"""
            )

        if max_calls_per_min:
            warnings.warn(
                "max_calls_per_min is deprecated and will not be used. Use LangChain's `InMemoryRateLimiter` instead",
                DeprecationWarning,
            )

    async def estimate_token_cost(
        self,
        tiktoken_model_name: str,
        prompts: List[str],
        example_responses: List[str] = None,
        response_sample_size: int = 30,
        system_prompt: str = "You are a helpful assistant",
        count: int = 25,
    ) -> Dict[str, float]:
        """
        Estimates the token cost for a given list of prompts and (optionally) example responses.
        Note: This method is only compatible with GPT models. Cost-per-token values are as of
        10/21/2024.

        Parameters
        ----------
        tiktoken_model_name: str
           The name of the OpenAI model to use for token counting.

        prompts : list of strings
           A list of prompts

        example_responses : list of strings, default=None
           A list of example responses. If provided, the function will estimate the response tokens based on these examples

        response_sample_size : int, default=30.
           The number of responses to generate for cost estimation if `example_responses` is not provided.

        system_prompt : str, default="You are a helpful assistant."
           Specifies the system prompt used when generating LLM responses.

        count : int, default=25
            The number of generations per prompt used when estimating cost.

        Returns
        -------
        dict
           A dictionary containing the estimated token costs, including prompt token cost, completion token cost,
           and total token cost.
        """
        # TODO: Add token costs for other models
        # TODO: Scrape rather than hard-code costs.
        print(
            f"Token costs were last updated on {self.token_cost_date} and may have changed since then."
        )
        assert (
            tiktoken_model_name in self.cost_mapping.keys()
        ), f"Only {list(self.cost_mapping.keys())} are supported"

        print(f"Estimating cost based on {count} generations per prompt...")

        if example_responses is None:
            print("Generating sample of responses for cost estimation...")
            prompts = list(prompts)
            sampled_prompts = random.sample(
                prompts, min(response_sample_size, len(prompts))
            )  # nosec - bandit thinks this insecure use of random.sample could be used in a crypto context
            generation = await self.generate_responses(sampled_prompts, count=1)
            example_responses = generation["data"]["response"]

        # Get input token counts
        prompt_token_counts = [
            self._num_tokens_from_messages(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=tiktoken_model_name,
            )
            for user_prompt in prompts
        ]
        total_prompt_tokens = sum(prompt_token_counts) * count

        # Estimate output token counts
        example_response_tokens = [
            self._num_tokens_from_messages(
                [{"role": "assistant", "content": response}],
                model=tiktoken_model_name,
                prompt=False,
            )
            for response in example_responses
        ]
        estimated_total_response_tokens = (
            len(prompts) * np.mean(example_response_tokens) * count
        )

        # calculate costs
        model_cost = self.cost_mapping.get(
            tiktoken_model_name, {"input": 0, "output": 0}
        )
        estimated_prompt_token_cost = total_prompt_tokens * model_cost["input"]
        estimated_completion_token_cost = (
            estimated_total_response_tokens * model_cost["output"]
        )
        estimated_total_token_cost = (
            estimated_prompt_token_cost + estimated_completion_token_cost
        )

        results = {
            "Estimated Prompt Token Cost (USD)": estimated_prompt_token_cost,
            "Estimated Completion Token Cost (USD)": estimated_completion_token_cost,
            "Estimated Total Token Cost (USD)": estimated_total_token_cost,
        }
        return results

    async def generate_responses(
        self,
        prompts: List[str],
        system_prompt: str = "You are a helpful assistant.",
        count: int = 25,
    ) -> Dict[str, Any]:
        """
        Generates evaluation dataset from a provided set of prompts. For each prompt,
        `self.count` responses are generated.

        Parameters
        ----------
        prompts : list of strings
            List of prompts from which LLM responses will be generated

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        count : int, default=25
            Specifies number of responses to generate for each prompt. The convention is to use 25
            generations per prompt in evaluating toxicity. See, for example DecodingTrust (https://arxiv.org/abs//2306.11698)
            or Gehman et al., 2020 (https://aclanthology.org/2020.findings-emnlp.301/).

        Returns
        -------
        dict
            A dictionary with two keys: 'data' and 'metadata'.

            'data' : dict
                A dictionary containing the prompts and responses.

                'prompt' : list
                    A list of prompts.
                'response' : list
                    A list of responses corresponding to the prompts.

            'metadata' : dict
                A dictionary containing metadata about the generation process.

                'non_completion_rate' : float
                    The rate at which the generation process did not complete.
                'temperature' : float
                    The temperature parameter used in the generation process.
                'count' : int
                    The count of prompts used in the generation process.
                'system_prompt' : str
                    The system prompt used for generating responses
        """
        assert isinstance(
            self.llm, langchain_core.language_models.chat_models.BaseChatModel
        ), """
            langchain_llm must be an instance of langchain_core.language_models.chat_models.BaseChatModel
        """
        assert all(
            isinstance(prompt, str) for prompt in prompts
        ), "If using custom prompts, please ensure `prompts` is of type list[str]"

        if self.use_n_param:
            warnings.warn(N_PARAM_WARNING)
            if not ((count > 1) and (hasattr(self.llm, "n"))):
                self.use_n_param = False

        print(f"Generating {count} responses per prompt...")
        if self.llm.temperature == 0:
            assert count == 1, "temperature must be greater than 0 if count > 1"
        self._update_count(count)
        self.system_message = SystemMessage(system_prompt)

        tasks, duplicated_prompts = self._create_tasks(prompts=prompts)
        response_lists = await asyncio.gather(*tasks)

        responses = []
        for response in response_lists:
            responses.extend(response)

        print("Responses successfully generated!")
        return {
            "data": {
                "prompt": self._enforce_strings(duplicated_prompts),
                "response": self._enforce_strings(responses),
            },
            "metadata": {
                "non_completion_rate": self._calc_noncompletion_rate(responses),
                "system_prompt": system_prompt,
                "temperature": self.llm.temperature,
                "count": self.count,
            },
        }

    def _update_count(self, count: int) -> None:
        """Updates self.count parameter and self.llm as necessary"""
        self.count = count
        if self.use_n_param:
            self.llm.n = count
        elif hasattr(self.llm, "n"):
            self.llm.n = 1

    def _create_tasks(
        self,
        prompts: List[str],
    ) -> Tuple[List[Any], List[str]]:
        """
        Creates a list of async tasks and returns duplicated prompt list
        with each prompt duplicated `count` times
        """
        duplicated_prompts = [
            prompt for prompt, i in itertools.product(prompts, range(self.count))
        ]
        if self.use_n_param:
            try:
                tasks = [
                    self._async_api_call(prompt=prompt, count=self.count)
                    for prompt in prompts
                ]
            except ValueError:
                self.use_n_param = False
                self.llm.n = 1
        if not self.use_n_param:
            tasks = [
                self._async_api_call(prompt=prompt, count=1)
                for prompt in duplicated_prompts
            ]
        return tasks, duplicated_prompts

    async def _async_api_call(self, prompt: str, count: int = 1) -> List[Any]:
        """Generates responses asynchronously using a BaseLanguageModel object"""
        messages = [self.system_message, HumanMessage(prompt)]
        try:
            result = await self.llm.agenerate([messages])
            generations = [result.generations[0][i].text for i in range(count)]
            if len(generations) != count:
                raise ValueError("Incorrect number of generations")
            return generations
        except Exception as err:
            if self.suppressed_exceptions is not None:
                if isinstance(self.suppressed_exceptions, Dict):
                    if isinstance(err, tuple(self.suppressed_exceptions.keys())):
                        return [self.suppressed_exceptions.get(type(err))] * count
                elif isinstance(err, self.suppressed_exceptions):
                    return [FAILURE_MESSAGE] * count
            raise err

    def _calc_noncompletion_rate(self, responses: List[str]) -> float:
        """Compute noncompletion rate"""
        if isinstance(self.suppressed_exceptions, Dict):
            non_completion_rate = len(
                [
                    r
                    for r in responses
                    if any(r == value for value in self.suppressed_exceptions.values())
                    or r == FAILURE_MESSAGE
                ]
            ) / len(responses)
        else:
            non_completion_rate = len(
                [r for r in responses if r == FAILURE_MESSAGE]
            ) / len(responses)
        return non_completion_rate

    @staticmethod
    def _valid_exceptions(
        exceptions: Union[Tuple[BaseException], BaseException],
    ) -> bool:
        """Returns true if exceptions is a subclass of BaseException or a tuple of  subclasses of BaseException"""
        if exceptions is None:
            return True
        else:
            try:
                if isinstance(exceptions, tuple) and all(
                    issubclass(item, BaseException) for item in exceptions
                ):
                    return True
                elif issubclass(exceptions, BaseException):
                    return True
                else:
                    return False
            except Exception:
                return False

    @staticmethod
    def _enforce_strings(texts: List[Any]) -> List[str]:
        """Enforce that all outputs are strings"""
        return [str(r) for r in texts]

    @staticmethod
    def _num_tokens_from_messages(
        messages: List[Dict[str, str]], model: str, prompt: bool = True
    ) -> int:
        """
        Returns the number of tokens used by a list of messages.

        Note : This code is adapted from the `openai-cookbook` GitHub repository.
        Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        model_data = {
            "gpt-3.5-turbo-0301": (4, 1),
            "gpt-3.5-turbo-0613": (3, 1),
            "gpt-3.5-turbo-16k-0613": (3, 1),
            "gpt-4-0314": (3, 1),
            "gpt-4-32k-0314": (3, 1),
            "gpt-4-0613": (3, 1),
            "gpt-4-32k-0613": (3, 1),
        }
        if model not in model_data:
            if "gpt-3.5-turbo" in model:
                print(
                    "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
                )
                model = "gpt-3.5-turbo-0613"
            elif "gpt-4" in model:
                print(
                    "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
                )
                model = "gpt-4-0613"
            else:
                raise NotImplementedError(
                    f"""cost_estimator() is not implemented for model {model}."""
                )
        tokens_per_message, tokens_per_name = model_data[model]
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in messages:
            if prompt:
                num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
        if prompt:
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        elif not prompt:
            num_tokens += -1
        return num_tokens
