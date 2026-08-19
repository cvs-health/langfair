# Copyright 2026 CVS Health and/or one of its affiliates
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

import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

from langfair.constants.cost_data import FAILURE_MESSAGE
from langfair.metrics.counterfactual import CounterfactualMetrics
from langfair.metrics.stereotype import StereotypeMetrics
from langfair.metrics.toxicity import ToxicityMetrics
from langfair.utils.display import stop_progress_bar

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


class _BaseEval:
    """
    Shared logic for `AutoEval` and `SelfEval`: fairness-through-unawareness (FTU)
    checking, toxicity/stereotype/counterfactual metric computation, results storage,
    and results printing/exporting. Subclasses are responsible for supplying
    `self.prompts`, `self.responses`, `self.counterfactual_responses` (populated
    either by LLM generation or by user-provided responses) and must set
    `self.cf_generator_object` (a `CounterfactualGenerator`), which the FTU check
    relies on for parsing.
    """

    def __init__(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        metrics: MetricTypes = None,
        counterfactual_transformer: str = "all-MiniLM-L6-v2",
        counterfactual_sentiment_classifier: str = "vader",
        toxicity_device: str = "cpu",
        neutralize_tokens: bool = True,
        _stereotype_classifier_model: str = "wu981526092/Sentence-Level-Stereotype-Detector",
    ) -> None:
        self.prompts = self._validate_list_type(prompts)
        self.responses = self._validate_list_type(responses)
        self.counterfactual_responses = None
        self.metrics = self._validate_metrics(metrics)
        self.toxicity_device = toxicity_device
        self.neutralize_tokens = neutralize_tokens
        self._stereotype_classifier_model = _stereotype_classifier_model
        self.results = {"metrics": {}, "data": {}}
        self.counterfactual_transformer = counterfactual_transformer
        self.counterfactual_sentiment_classifier = counterfactual_sentiment_classifier
        self.progress_bar = None
        self.progress_task = None
        self.suppressed_exceptions = None

    def _check_ftu(
        self, show_progress_bars: bool, step_num: int = 1
    ) -> Tuple[Dict[str, int], int]:
        """
        Check for fairness through unawareness by parsing prompts for
        protected attribute words. Returns a dictionary containing counts of
        prompts with attribute words, by attribute, along with the total count.
        """
        if show_progress_bars:
            self.progress_bar.add_task(
                f"[No Progress Bar]Step {step_num}: Fairness Through Unawareness Check"
            )
            self.progress_bar.add_task(
                "[No Progress Bar]------------------------------------------"
            )
        else:
            print(f"Step {step_num}: Fairness Through Unawareness Check")
            print("------------------------------------------")
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
            if show_progress_bars:
                self.progress_bar.add_task(
                    f"[No Progress Bar]Number of prompts containing {attribute} words: {protected_words[attribute]}"
                )
            else:
                print(
                    f"""Number of prompts containing {attribute} words: {protected_words[attribute]}"""
                )
        return protected_words, total_protected_words

    def _evaluate_toxicity(self, show_progress_bars: bool, step_num: int = 4) -> None:
        """Calculate toxicity metrics on `self.prompts` and `self.responses`."""
        if show_progress_bars:
            self.progress_bar.add_task(
                f"[No Progress Bar]\nStep {step_num}: Evaluate Toxicity Metrics"
            )
            self.progress_bar.add_task(
                "[No Progress Bar]---------------------------------"
            )
        else:
            print(f"\nStep {step_num}: Evaluate Toxicity Metrics")
            print("---------------------------------")
        toxicity_object = ToxicityMetrics(device=self.toxicity_device)
        toxicity_results = toxicity_object.evaluate(
            prompts=list(self.prompts),
            responses=list(self.responses),
            return_data=True,
            show_progress_bars=show_progress_bars,
            existing_progress_bar=self.progress_bar,
        )
        self.results["metrics"]["Toxicity"] = toxicity_results["metrics"]

        del toxicity_results["data"]["response"], toxicity_results["data"]["prompt"]
        self.toxicity_scores = toxicity_results["data"]
        del toxicity_results

    def _evaluate_stereotype(
        self,
        protected_words: Dict[str, int],
        show_progress_bars: bool,
        step_num: int = 5,
    ) -> None:
        """Calculate stereotype metrics on `self.prompts` and `self.responses`."""
        if show_progress_bars:
            self.progress_bar.add_task(
                f"[No Progress Bar]\nStep {step_num}: Evaluate Stereotype Metrics"
            )
            self.progress_bar.add_task(
                "[No Progress Bar]-----------------------------------"
            )
        else:
            print(f"\nStep {step_num}: Evaluate Stereotype Metrics")
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
            show_progress_bars=show_progress_bars,
            existing_progress_bar=self.progress_bar,
        )
        self.results["metrics"]["Stereotype"] = stereotype_results["metrics"]

        del stereotype_results["data"]["response"], stereotype_results["data"]["prompt"]
        self.stereotype_scores = stereotype_results["data"]
        del stereotype_results

    def _evaluate_counterfactual(
        self,
        protected_words: Dict[str, int],
        total_protected_words: int,
        show_progress_bars: bool,
        step_num: int = 6,
    ) -> None:
        """
        Calculate counterfactual metrics from `self.counterfactual_responses`
        (if FTU not satisfied and counterfactual metrics requested).
        """
        if total_protected_words > 0 and "counterfactual" in self.metrics:
            if show_progress_bars:
                self.progress_bar.start()
                self.progress_bar.add_task(
                    f"[No Progress Bar]\nStep {step_num}: Evaluate Counterfactual Metrics"
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]---------------------------------------"
                )
            else:
                print(f"\nStep {step_num}: Evaluate Counterfactual Metrics")
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
                if (
                    protected_words[attribute] > 0
                    and attribute in self.counterfactual_responses
                ):
                    if show_progress_bars:
                        cf_task = self.progress_bar.add_task(
                            f"Computing counterfactual metrics for {attribute}...",
                            total=protected_words[attribute],
                        )
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
                            show_progress_bars=False,
                        )
                        self.results["metrics"]["Counterfactual"][
                            f"{group1}-{group2}"
                        ] = cf_group_results["metrics"]
                        self.counterfactual_data[f"{group1}-{group2}"] = (
                            cf_group_results["data"]
                        )
                    if show_progress_bars:
                        self.progress_bar.update(
                            cf_task, advance=protected_words[attribute]
                        )
                    time.sleep(0.1)
        else:
            if show_progress_bars:
                self.progress_bar.start()
                self.progress_bar.add_task(
                    f"[No Progress Bar]\n(Skipping) Step {step_num}: Evaluate Counterfactual Metrics"
                )
                self.progress_bar.add_task(
                    "[No Progress Bar]--------------------------------------------------"
                )
            else:
                print(f"\n(Skipping) Step {step_num}: Evaluate Counterfactual Metrics")
                print("--------------------------------------------------")
            # Initialize empty counterfactual_data when metrics are skipped
            self.counterfactual_data = {}

    def _finalize(self, return_data: bool, show_progress_bars: bool) -> Dict[str, Any]:
        """Attach response-level data if requested and return `self.results`."""
        if return_data:
            self.results["data"]["Toxicity"] = self.toxicity_data
            self.results["data"]["Stereotype"] = self.stereotype_data
            self.results["data"]["Counterfactual"] = self.counterfactual_data

        if show_progress_bars and self.progress_bar:
            stop_progress_bar(self.progress_bar)
        else:
            print("Evaluation complete.")

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
        se = self.suppressed_exceptions
        if isinstance(se, Dict):
            failure_messages = set(se.values())
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
