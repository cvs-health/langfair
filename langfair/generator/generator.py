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
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import langchain_core
import tiktoken
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from rich.progress import Progress

from langfair.constants.cost_data import COST_MAPPING, FAILURE_MESSAGE, TOKEN_COST_DATE
from langfair.utils.display import (
    start_progress_bar,
    stop_progress_bar,
)

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
        self.progress_bar = None
        self.progress_task = None
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
                stacklevel=2,
            )

    async def estimate_token_cost(
        self,
        tiktoken_model_name: str,
        prompts: List[str],
        example_responses: List[str] = None,
        response_sample_size: int = 30,
        system_prompt: str = "You are a helpful assistant",
        count: int = 25,
        show_progress_bars: bool = True,
        existing_progress_bar: Optional[Progress] = None,
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

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating and scoring responses

        existing_progress_bar : rich.progress.Progress, default=None
            If provided, the progress bar will be updated with the existing progress bar.

        Returns
        -------
        dict
           A dictionary containing the estimated token costs, including prompt token cost, completion token cost,
           and total token cost.
        """
        warnings.warn(
            "estimate_token_cost method has been deprecated as of v0.8.0 and will not be used",
            DeprecationWarning,
            stacklevel=2,
        )
        pass

    async def generate_responses(
        self,
        prompts: List[str],
        system_prompt: str = "You are a helpful assistant.",
        count: int = 25,
        show_progress_bars: bool = True,
        existing_progress_bar: Optional[Progress] = None,
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

        show_progress_bars : bool, default=True
            If True, displays progress bars while generating responses

        existing_progress_bar : rich.progress.Progress, default=None
            If provided, the progress bar will be updated with the existing progress bar.

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
        assert all(isinstance(prompt, str) for prompt in prompts), (
            "If using custom prompts, please ensure `prompts` is of type list[str]"
        )

        if self.use_n_param:
            warnings.warn(N_PARAM_WARNING)
            if not ((count > 1) and (hasattr(self.llm, "n"))):
                self.use_n_param = False

        if self.llm.temperature == 0:
            assert count == 1, "temperature must be greater than 0 if count > 1"
        self._update_count(count)
        self.system_message = SystemMessage(system_prompt)
                
        total = len(prompts) * self.count
        if show_progress_bars:
            self.progress_bar = start_progress_bar(existing_progress_bar)
            self.progress_task = self.progress_bar.add_task(
                f"Generating {self.count} responses per prompt...",
                total=total,
            )

        try:
            tasks, duplicated_prompts = self._create_tasks(prompts=prompts)
            response_lists = await asyncio.gather(*tasks)
        except Exception as e:
            stop_progress_bar(self.progress_bar)
            raise e

        if self.progress_bar:
            self.progress_bar.update(self.progress_task, completed=total)
        stop_progress_bar(self.progress_bar)
        responses = []
        for response in response_lists:
            responses.extend(response)

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
            if self.progress_bar:
                for _ in range(count):
                    self.progress_bar.update(self.progress_task, advance=1)
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
