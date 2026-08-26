# Copyright 2024 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

from langfair.auto.base import (
    DefaultMetrics,
    MetricTypes,
    Protected_Attributes,
    _BaseEval,
)
from langfair.generator import CounterfactualGenerator, ResponseGenerator
from langfair.utils.display import start_progress_bar

__all__ = ["AutoEval", "DefaultMetrics", "MetricTypes", "Protected_Attributes"]


class AutoEval(_BaseEval):
    def __init__(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        langchain_llm: Any = None,
        suppressed_exceptions: Optional[
            Union[Tuple[BaseException], BaseException, Dict[BaseException, str]]
        ] = None,
        use_n_param: bool = False,
        metrics: MetricTypes = None,
        counterfactual_transformer: str = "all-MiniLM-L6-v2",
        counterfactual_sentiment_classifier: str = "vader",
        toxicity_device: str = "cpu",
        neutralize_tokens: str = True,
        max_calls_per_min: Optional[int] = None,
        _stereotype_classifier_model: str = "wu981526092/Sentence-Level-Stereotype-Detector",
    ) -> None:
        """
        This class calculates all toxicity, stereotype, and counterfactual metrics support by langfair

        Parameters
        ----------
        prompts : list of strings or DataFrame of strings
            A list of input prompts for the model.

        responses : list of strings or DataFrame of strings, default is None
            A list of generated output from an LLM. If not available, responses are generated using the model.

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

        metrics : dict or list of str, default option compute all supported metrics.
            Specifies which metrics to evaluate.

        counterfactual_transformer: str, default="all-MiniLM-L6-v2"
            Specifies which huggingface sentence transformer to use when computing cosine distance. See
            https://huggingface.co/sentence-transformers?sort_models=likes#models
            for more information. The recommended sentence transformer is 'all-MiniLM-L6-v2'. User can also specify a local path to a model.

        counterfactual_sentiment_classifier: str, default="vader"
            Specifies the sentiment classifier to use for counterfactual sentiment bias calculation.

        toxicity_device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that toxicity classifiers use for prediction. Set to "cuda" for classifiers to be able
            to leverage the GPU. Currently, 'detoxify_unbiased' and 'detoxify_original' will use this parameter.

        neutralize_tokens: boolean, default=True
            An indicator attribute to use masking for the computation of Blue and RougeL metrics. If True, counterfactual
            responses are masked using `CounterfactualGenerator.neutralize_tokens` method before computing the aforementioned metrics.

        max_calls_per_min : int, default=None
            [Deprecated] Use LangChain's InMemoryRateLimiter instead.
        """
        super().__init__(
            prompts=prompts,
            responses=responses,
            metrics=metrics,
            counterfactual_transformer=counterfactual_transformer,
            counterfactual_sentiment_classifier=counterfactual_sentiment_classifier,
            toxicity_device=toxicity_device,
            neutralize_tokens=neutralize_tokens,
            _stereotype_classifier_model=_stereotype_classifier_model,
        )
        self.langchain_llm = langchain_llm
        self.use_n_param = use_n_param
        self.suppressed_exceptions = suppressed_exceptions

        self.cf_generator_object = CounterfactualGenerator(
            langchain_llm=langchain_llm,
            max_calls_per_min=max_calls_per_min,
            suppressed_exceptions=suppressed_exceptions,
            use_n_param=use_n_param,
        )
        self.generator_object = ResponseGenerator(
            langchain_llm=langchain_llm,
            max_calls_per_min=max_calls_per_min,
            suppressed_exceptions=suppressed_exceptions,
            use_n_param=use_n_param,
        )

    async def evaluate(
        self,
        count: int = 25,
        metrics: MetricTypes = None,
        return_data: bool = False,
        show_progress_bars: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all the metrics based on the provided data.

        Parameters
        ----------
        count : int, default=25
            Specifies number of responses to generate for each prompt. The convention is to use 25
            generations per prompt in evaluating toxicity. See, for example DecodingTrust (https://arxiv.org/abs//2306.11698)
            or Gehman et al., 2020 (https://aclanthology.org/2020.findings-emnlp.301/).

        metrics : dict or list of str, optional
            Specifies which metrics to evaluate. If None, computes all supported metrics.

        return_data : bool, default=False
            Indicates whether to include response-level scores in results dictionary returned by this method.

        Returns
        -------
        dict
            A dictionary containing values of toxicity, stereotype, and counterfactual metrics and, optionally,
            response-level scores.
        """
        if metrics is not None:
            self.metrics = self._validate_metrics(metrics)

        if show_progress_bars:
            self.progress_bar = start_progress_bar()

        # 1. Check for Fairness Through Unawareness FTU
        protected_words, total_protected_words = self._check_ftu(show_progress_bars)

        if total_protected_words > 0:
            if show_progress_bars:
                self.progress_bar.add_task(
                    "[No Progress Bar]FTU is not met. Counterfactual assessment will be conducted."
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]\nStep 2: Generate Counterfactual Datasets"
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]----------------------------------------"
                )
            else:
                print(
                    "Fairness through unawareness is not satisfied. Counterfactual assessments will be included."
                )
                print("\nStep 2: Generate Counterfactual Datasets")
                print("----------------------------------------")
            # 2. Generate CF responses for race (if race FTU not satisfied) and gender (if gender FTU not satisfied)
            if (self.counterfactual_responses is None) and (
                "counterfactual" in self.metrics
            ):
                self.counterfactual_responses = {}
                self.counterfactual_response_metadata = {}
                for attribute in protected_words.keys():
                    if protected_words[attribute] > 0:
                        try:
                            self.counterfactual_responses[
                                attribute
                            ] = await self.cf_generator_object.generate_responses(
                                count=count,
                                prompts=self.prompts,
                                attribute=attribute,
                                show_progress_bars=show_progress_bars,
                                existing_progress_bar=self.progress_bar,
                            )
                        except AssertionError as e:
                            # Handle case where prompts don't contain the specific attribute words
                            print(
                                f"Warning: Could not generate counterfactual responses for {attribute}: {e}"
                            )
                            # Remove this attribute from protected_words to prevent KeyError later
                            protected_words[attribute] = 0
        else:
            if show_progress_bars:
                self.progress_bar.add_task(
                    "[No Progress Bar]FTU is satisfied. Counterfactual assessment will be skipped."
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]\n\033[1m(Skipping) Step 2: Generate Counterfactual Dataset\033[0m"
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]--------------------------------------------------"
                )
            else:
                print("FTU is satisfied. Counterfactual assessment will be skipped.")
                print(
                    "\n\033[1m(Skipping) Step 2: Generate Counterfactual Dataset\033[0m"
                )
                print("--------------------------------------------------")

        # 3. Generate responses for toxicity and stereotype evaluation (if responses not provided)
        if self.responses is None:
            if show_progress_bars:
                self.progress_bar.add_task(
                    "[No Progress Bar]\nStep 3: Generating Model Responses"
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]----------------------------------"
                )
            else:
                print("\nStep 3: Generating Model Responses")
                print("----------------------------------")
            dataset = await self.generator_object.generate_responses(
                prompts=self.prompts,
                count=count,
                show_progress_bars=show_progress_bars,
                existing_progress_bar=self.progress_bar,
            )
            self.prompts = dataset["data"]["prompt"]
            self.responses = dataset["data"]["response"]
        else:
            if show_progress_bars:
                self.progress_bar.add_task(
                    "[No Progress Bar]\n(Skipping) Step 3: Generating Model Responses"
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]---------------------------------------------"
                )
            else:
                print("\n(Skipping) Step 3: Generating Model Responses")
                print("---------------------------------------------")

        # 4. Calculate toxicity metrics
        self._evaluate_toxicity(show_progress_bars)

        # 5. Calculate stereotype metrics
        self._evaluate_stereotype(protected_words, show_progress_bars)

        # 6. Calculate CF metrics (if FTU not satisfied and counterfactual metrics requested)
        self._evaluate_counterfactual(
            protected_words, total_protected_words, show_progress_bars
        )

        return self._finalize(return_data, show_progress_bars)
