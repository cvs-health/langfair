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

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

from langfair.constants.cost_data import FAILURE_MESSAGE
from langfair.generator import CounterfactualGenerator, ResponseGenerator
from langfair.metrics.counterfactual import CounterfactualMetrics
from langfair.metrics.stereotype import StereotypeMetrics
from langfair.metrics.toxicity import ToxicityMetrics

MetricTypes = Union[None, list, dict]
DefaultMetrics = {
    "counterfactual": ["Cosine", "Rougel", "Bleu", "Sentiment Bias"],
    "stereotype": [
        "Stereotype Association",
        "Cooccurrence Bias",
        "Stereotype Classifier",
    ],
    "toxicity": ["Toxic Fraction", "Expected Maximum Toxicity", "Toxicity Probability"],
}
Protected_Attributes = {
    "race": ["white", "black", "asian", "hispanic"],
    "gender": ["male", "female"],
}


class AutoEval:
    def __init__(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        langchain_llm: Any = None,
        suppressed_exceptions: Optional[
            Union[Tuple[BaseException], BaseException, Dict[BaseException, str]]
        ] = None,
        use_n_param: bool = True,
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
        self.prompts = self._validate_list_type(prompts)
        self.responses = self._validate_list_type(responses)
        self.counterfactual_responses = None
        self.langchain_llm = langchain_llm
        self.metrics = self._validate_metrics(metrics)
        self.use_n_param = use_n_param
        self.toxicity_device = toxicity_device
        self.neutralize_tokens = neutralize_tokens
        self._stereotype_classifier_model = _stereotype_classifier_model
        self.results = {"metrics": {}, "data": {}}
        self.counterfactual_transformer = counterfactual_transformer
        self.counterfactual_sentiment_classifier = counterfactual_sentiment_classifier

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
        self, count: int = 25, metrics: MetricTypes = None, return_data: bool = False
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

        print("\033[1mStep 1: Fairness Through Unawareness Check\033[0m")
        print("------------------------------------------")
        # 1. Check for Fairness Through Unawareness FTU
        # Parse prompts for protected attribute words
        protected_words = {"race": 0, "gender": 0}
        total_protected_words = 0
        for attribute in protected_words.keys():
            col = self.cf_generator_object.parse_texts(
                texts=self.prompts, attribute=attribute
            )
            protected_words[attribute] = sum(
                [1 if len(col_item) > 0 else 0 for col_item in col]
            )
            total_protected_words += protected_words[attribute]
            print(
                f"""Number of prompts containing {attribute} words: {protected_words[attribute]}"""
            )

        if total_protected_words > 0:
            print(
                "Fairness through unawareness is not satisfied. Toxicity, stereotype, and counterfactual fairness assessments will be conducted."
            )
            print("\n\033[1mStep 2: Generate Counterfactual Dataset\033[0m")
            print("---------------------------------------")
            # 2. Generate CF responses for race (if race FTU not satisfied) and gender (if gender FTU not satisfied)
            if (self.counterfactual_responses is None) and (
                "counterfactual" in self.metrics
            ):
                self.counterfactual_responses = {}
                self.counterfactual_response_metadata = {}
                for attribute in protected_words.keys():
                    if protected_words[attribute] > 0:
                        self.counterfactual_responses[
                            attribute
                        ] = await self.cf_generator_object.generate_responses(
                            count=count, prompts=self.prompts, attribute=attribute
                        )
        else:
            print(
                "Fairness through unawareness is satisfied. Toxicity and stereotype assessments will be conducted."
            )
            print("\n\033[1m(Skipping) Step 2: Generate Counterfactual Dataset\033[0m")
            print("--------------------------------------------------")

        # 3. Generate responses for toxicity and stereotype evaluation (if responses not provided)
        if self.responses is None:
            print("\n\033[1mStep 3: Generating Model Responses\033[0m")
            print("----------------------------------")
            dataset = await self.generator_object.generate_responses(
                prompts=self.prompts,
                count=count,
            )
            self.prompts = dataset["data"]["prompt"]
            self.responses = dataset["data"]["response"]
        else:
            print("\n\033[1m(Skipping) Step 3: Generating Model Responses\033[0m")
            print("---------------------------------------------")

        # 4. Calculate toxicity metrics
        print("\n\033[1mStep 4: Evaluate Toxicity Metrics\033[0m")
        print("---------------------------------")
        toxicity_object = ToxicityMetrics(device=self.toxicity_device)
        toxicity_results = toxicity_object.evaluate(
            prompts=list(self.prompts), responses=list(self.responses), return_data=True
        )
        self.results["metrics"]["Toxicity"] = toxicity_results["metrics"]

        del toxicity_results["data"]["response"], toxicity_results["data"]["prompt"]
        self.toxicity_scores = toxicity_results["data"]
        del toxicity_results

        # 5. Calculate stereotype metrics
        print("\n\033[1mStep 5: Evaluate Stereotype Metrics\033[0m")
        print("-----------------------------------")
        attributes = [
            attribute
            for attribute in protected_words.keys()
            if protected_words[attribute] > 0
        ]
        stereotype_object = StereotypeMetrics(
            _classifier_model=self._stereotype_classifier_model
        )
        stereotype_results = stereotype_object.evaluate(
            prompts=list(self.prompts),
            responses=list(self.responses),
            return_data=True,
            categories=attributes,
        )
        self.results["metrics"]["Stereotype"] = stereotype_results["metrics"]

        del stereotype_results["data"]["response"], stereotype_results["data"]["prompt"]
        self.stereotype_scores = stereotype_results["data"]
        del stereotype_results

        # 6. Calculate CF metrics (if FTU not satisfied)
        if total_protected_words > 0:
            print("\n\033[1mStep 6: Evaluate Counterfactual Metrics\033[0m")
            print("---------------------------------------")
            print("Evaluating metrics...")
            self.results["metrics"]["Counterfactual"] = {}
            self.counterfactual_data = {}
            counterfactual_object = CounterfactualMetrics(
                neutralize_tokens=self.neutralize_tokens,
                transformer=self.counterfactual_transformer,
                sentiment_classifier=self.counterfactual_sentiment_classifier,
            )
            for attribute in Protected_Attributes.keys():
                if protected_words[attribute] > 0:
                    for group1, group2 in combinations(
                        Protected_Attributes[attribute], 2
                    ):
                        group1_response = self.counterfactual_responses[attribute][
                            "data"
                        ][group1 + "_response"]
                        group2_response = self.counterfactual_responses[attribute][
                            "data"
                        ][group2 + "_response"]
                        successful_response_index = self._get_success_indices(
                            group1_response=group1_response,
                            group2_response=group2_response,
                        )
                        cf_group_results = counterfactual_object.evaluate(
                            texts1=[
                                group1_response[i] for i in successful_response_index
                            ],
                            texts2=[
                                group2_response[i] for i in successful_response_index
                            ],
                            attribute=attribute,
                            return_data=True,
                        )
                        self.results["metrics"]["Counterfactual"][
                            f"{group1}-{group2}"
                        ] = cf_group_results["metrics"]
                        self.counterfactual_data[f"{group1}-{group2}"] = (
                            cf_group_results["data"]
                        )
        else:
            print("\n\033[1m(Skipping) Step 6: Evaluate Counterfactual Metrics\033[0m")
            print("--------------------------------------------------")

        if return_data:
            self.results["data"]["Toxicity"] = self.toxicity_data
            self.results["data"]["Stereotype"] = self.stereotype_data
            self.results["data"]["Counterfactual"] = self.counterfactual_data

        return self.results

    @property
    def toxicity_data(self):
        self.toxicity_scores["prompt"] = self.prompts
        self.toxicity_scores["response"] = self.responses
        return self.toxicity_scores

    @property
    def stereotype_data(self):
        self.stereotype_scores["prompt"] = self.prompts
        self.stereotype_scores["response"] = self.responses
        return self.stereotype_scores

    def print_results(self) -> None:
        """
        Print the evaluate metrics values in the desired format.
        """
        result_list = self._create_result_list()
        print("".join(result_list))

    def export_results(self, file_name: str = "results.txt") -> None:
        """
        Export the evaluated metrics values in a text file.

        Parameters
        ----------
        file_name : str, Default = "results.txt"
            Name of the .txt file.
        """
        result_list = self._create_result_list(bold_headings=False)

        with open(file_name, "w+") as file:
            # Writing data to a file
            file.writelines(result_list)

    def _get_success_indices(
        self, group1_response: List[str], group2_response: List[str]
    ) -> List[any]:
        se = self.cf_generator_object.suppressed_exceptions
        if isinstance(se, Dict):
            failure_messages = set(self.suppressed_exceptions.values())
            failure_messages.add(FAILURE_MESSAGE)
            successful_response_index = [
                i
                for i in range(len(group1_response))
                if group1_response[i] not in failure_messages
                and group2_response[i] not in failure_messages
            ]
        else:
            successful_response_index = [
                i
                for i in range(len(group1_response))
                if group1_response[i] != FAILURE_MESSAGE
                and group2_response[i] != FAILURE_MESSAGE
            ]
        return successful_response_index

    def _create_result_list(self, bold_headings=True) -> List[str]:
        """Helper function for `print_results` method."""
        result_list = []
        start_heading, end_heading = "", ""
        if bold_headings:
            start_heading, end_heading = "\033[1m", "\033[0m"
        result_list.append(
            start_heading + "1. Toxicity Assessment" + end_heading + " \n"
        )
        for key in self.results["metrics"]["Toxicity"]:
            result_list.append(
                "- {:<40} {:1.4f} \n".format(
                    key, self.results["metrics"]["Toxicity"][key]
                )
            )

        result_list.append(
            start_heading + "2. Stereotype Assessment" + end_heading + " \n"
        )
        for key in self.results["metrics"]["Stereotype"]:
            tmp = "- {:<40} {:1.4f} \n"
            if self.results["metrics"]["Stereotype"][key] is None:
                tmp = "- {:<40} {} \n"
            result_list.append(
                tmp.format(key, self.results["metrics"]["Stereotype"][key])
            )

        if "Counterfactual" in self.results["metrics"]:
            result_list.append(
                start_heading + "3. Counterfactual Assessment" + end_heading + " \n"
            )
            tmp = ["{:<25}".format(" ")]
            for key in self.results["metrics"]["Counterfactual"]:
                tmp.append("{:<15}".format(key))
            tmp.append(" \n")
            result_list.append("".join(tmp))

            for metric_name in list(self.results["metrics"]["Counterfactual"].values())[
                0
            ]:
                tmp = ["- ", "{:<25}".format(metric_name)]
                for key in self.results["metrics"]["Counterfactual"]:
                    tmp.append(
                        "{:<15}".format(
                            "{:1.4f}".format(
                                self.results["metrics"]["Counterfactual"][key][
                                    metric_name
                                ]
                            )
                        )
                    )
                tmp.append(" \n")
                result_list.append("".join(tmp))
        return result_list

    def _validate_metrics(self, metrics: List[str]) -> None:
        """Validate that specified metrics are supported."""
        if metrics is None:
            metrics = DefaultMetrics
        elif isinstance(metrics, list):
            tmp = dict()
            for metric in metrics:
                if metric in DefaultMetrics:
                    tmp[metric] = DefaultMetrics[metric]
                else:
                    raise RuntimeError(
                        "If `metrics` is a list, it should be a subset of following list ['counterfactual', 'stereotype', 'toxicity']"
                    )
            metrics = tmp
        elif isinstance(metrics, dict):
            for key in metrics.keys():
                if key not in DefaultMetrics.keys():
                    raise KeyError("{} not found".format(key))
                self._check_list(
                    metrics[key], DefaultMetrics[key], "metrics['" + key + "']"
                )
        else:
            raise TypeError(
                "Attribute `metrics` should be a list of strings or a dictionary of list of strings"
            )
        return metrics

    @staticmethod
    def _validate_list_type(input_variable: List[str]) -> List[str]:
        """Validate inputs."""
        if isinstance(input_variable, list):
            if len(input_variable) == 0 or not isinstance(input_variable[0], str):
                raise RuntimeError(
                    "List {} should contain strings and can't be empty.".format(
                        input_variable
                    )
                )
        return input_variable

    @staticmethod
    def _check_list(list1: List[str], list2: List[str], error_tag: Any) -> None:
        """
        Check if list1 is a subset of list 2.
        """
        for list1_i in list1:
            if not isinstance(list1_i, str):
                raise TypeError(
                    "Type of list '{}' should be a string.".format(error_tag)
                )
            elif list1_i not in list2:
                raise RuntimeError(
                    "Provided '{}' metric is not an in-built langfair metric".format(
                        error_tag
                    )
                )
