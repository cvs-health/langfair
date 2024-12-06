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
        suppressed_exceptions: Optional[Tuple] = None,
        metrics: MetricTypes = None,
        toxicity_device: str = "cpu",
        neutralize_tokens: str = True,
        max_calls_per_min: Optional[int] = None,
    ) -> None:
        """
        This class calculates all toxicity, stereotype, and counterfactual metrics support by langfair

        Parameters
        ----------
        prompts : list of strings or DataFrame of strings
            A list of input prompts for the model.

        responses : list of strings or DataFrame of strings, default is None
            A list of generated output from an LLM. If not available, responses are generated using the model.

        langchain_llm : langchain llm object, default=None
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `langchain_llm` object.

        suppressed_exceptions : tuple, default=None
            Specifies which exceptions to handle as 'Unable to get response' rather than raising the
            exception

        metrics : dict or list of str, default option compute all supported metrics.
            Specifies which metrics to evaluate.

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
        self.toxicity_device = toxicity_device
        self.neutralize_tokens = neutralize_tokens
        self.results = {}

        self.cf_generator_object = CounterfactualGenerator(
            langchain_llm=langchain_llm,
            max_calls_per_min=max_calls_per_min,
            suppressed_exceptions=suppressed_exceptions,
        )
        self.generator_object = ResponseGenerator(
            langchain_llm=langchain_llm,
            max_calls_per_min=max_calls_per_min,
            suppressed_exceptions=suppressed_exceptions,
        )

    async def evaluate(
        self, metrics: MetricTypes = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all the metrics based on the provided data.

        Parameters
        ----------
        metrics : dict or list of str, optional
            Specifies which metrics to evaluate. If None, computes all supported metrics.

        Returns
        -------
        dict
            A dictionary containing values of toxicity, stereotype, and counterfactual metrics.
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
                        cf_generation = (
                            await self.cf_generator_object.generate_responses(
                                prompts=self.prompts, attribute=attribute
                            )
                        )
                        cf_response = {
                            k: v
                            for k, v in cf_generation.items()
                            if self.cf_generator_object.failure_message not in v
                        }
                        self.counterfactual_responses[attribute] = cf_response
                        self.counterfactual_response_metadata[attribute] = (
                            cf_generation["metadata"]
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
            dataset = await self.generator_object.generate_responses(self.prompts)
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
        self.results["Toxicity"], self.toxicity_data = (
            toxicity_results["metrics"],
            toxicity_results["data"],
        )

        # 5. Calculate stereotype metrics
        print("\n\033[1mStep 5: Evaluate Stereotype Metrics\033[0m")
        print("-----------------------------------")
        attributes = [
            attribute
            for attribute in protected_words.keys()
            if protected_words[attribute] > 0
        ]
        stereotype_object = StereotypeMetrics()
        stereotype_results = stereotype_object.evaluate(
            prompts=list(self.prompts),
            responses=list(self.responses),
            return_data=True,
            categories=attributes,
        )
        self.results["Stereotype"], self.stereotype_data = (
            stereotype_results["metrics"],
            stereotype_results["data"],
        )

        # 6. Calculate CF metrics (if FTU not satisfied)
        if total_protected_words > 0:
            print("\n\033[1mStep 6: Evaluate Counterfactual Metrics\033[0m")
            print("---------------------------------------")
            print("Evaluating metrics...")
            cf_results = {}
            counterfactual_object = CounterfactualMetrics(
                neutralize_tokens=self.neutralize_tokens
            )
            for attribute in Protected_Attributes.keys():
                if protected_words[attribute] > 0:
                    for group1, group2 in combinations(
                        Protected_Attributes[attribute], 2
                    ):
                        cf_results[f"{group1}-{group2}"] = (
                            counterfactual_object.evaluate(
                                texts1=self.counterfactual_responses[attribute]["data"][
                                    group1 + "_response"
                                ],
                                texts2=self.counterfactual_responses[attribute]["data"][
                                    group2 + "_response"
                                ],
                                attribute=attribute,
                            )
                        )
                    self.results["Counterfactual"] = cf_results
        else:
            print("\n\033[1m(Skipping) Step 6: Evaluate Counterfactual Metrics\033[0m")
            print("--------------------------------------------------")

        return self.results

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

    def _create_result_list(self, bold_headings=True) -> List[str]:
        """Helper function for `print_results` method."""
        result_list = []
        start_heading, end_heading = "", ""
        if bold_headings:
            start_heading, end_heading = "\033[1m", "\033[0m"
        result_list.append(
            start_heading + "1. Toxicity Assessment" + end_heading + " \n"
        )
        for key in self.results["Toxicity"]:
            result_list.append(
                "- {:<40} {:1.4f} \n".format(key, self.results["Toxicity"][key])
            )

        if "Stereotype" in self.results:
            result_list.append(
                start_heading + "2. Stereotype Assessment" + end_heading + " \n"
            )
            for key in self.results["Stereotype"]:
                tmp = "- {:<40} {:1.4f} \n"
                if self.results["Stereotype"][key] is None:
                    tmp = "- {:<40} {} \n"
                result_list.append(tmp.format(key, self.results["Stereotype"][key]))

        if "Counterfactual" in self.results:
            result_list.append(
                start_heading + "3. Counterfactual Assessment" + end_heading + " \n"
            )
            tmp = ["{:<25}".format(" ")]
            for key in self.results["Counterfactual"]:
                tmp.append("{:<15}".format(key))
            tmp.append(" \n")
            result_list.append("".join(tmp))

            for metric_name in list(self.results["Counterfactual"].values())[0]:
                tmp = ["- ", "{:<25}".format(metric_name)]
                for key in self.results["Counterfactual"]:
                    tmp.append(
                        "{:<15}".format(
                            "{:1.4f}".format(
                                self.results["Counterfactual"][key][metric_name]
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
