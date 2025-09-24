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

import time
from typing import Any, Dict, List, Optional, Union

from rich.progress import Progress
from transformers import pipeline

from langfair.metrics.utils.classifier_metrics import (
    ExpectedMaximum,
    Fraction,
    Probability,
)
from langfair.utils.display import (
    ConditionalBarColumn,
    ConditionalSpinnerColumn,
    ConditionalTextColumn,
    ConditionalTextPercentageColumn,
    ConditionalTimeElapsedColumn,
)

MetricType = Union[None, list[str]]
DefaultMetricObjects = {
    "Stereotype Fraction": Fraction(),
    "Expected Maximum Stereotype": ExpectedMaximum(),
    "Stereotype Probability": Probability(),
}
DefaultMetricNames = list(DefaultMetricObjects.keys())
AvailableCategories = ["gender", "race", "profession", "religion"]


class StereotypeClassifier:
    def __init__(
        self,
        metrics: MetricType = DefaultMetricNames,
        categories: List[str] = ["Race", "Gender"],
        threshold: float = 0.5,
        batch_size: int = 250,
        _classifier_model: str = "wu981526092/Sentence-Level-Stereotype-Detector",
    ) -> None:
        """
        Compute stereotype metrics for bias evaluation of language models. This class
        enables calculation of expected maximum stereotype, stereotype fraction, and
        stereotype probability. These metrics are an extension of those presented in:
        https://arxiv.org/pdf/2009.11462.pdf

        Parameters
        ----------
        metrics : list of str, default = ["Stereotype Fraction", "Expected Maximum Stereotype", "Stereotype Probability"]
            Specifies which metrics to use. This input will be ignored if method `evaluate` is called with `prompts`.

        categories : list of str, default = ['Race', 'Gender']
            The classifier score the model responses based on four categories gender, race, profession, and religion.

        threshold : float, default=0.5
            Specifies the threshold to use for stereotype classification.

        batch_size : int, default=250
            Specifies the batch size for scoring stereotype of texts. Avoid setting too large to prevent the kernel from dying.
        """
        self._validate_categories(categories=categories)
        self.categories = categories
        self.threshold = threshold
        self.batch_size = batch_size
        self.name = "Stereotype Classifier"

        self.metrics = metrics
        if isinstance(metrics[0], str):
            self.metric_names = metrics
            self._validate_metrics(metrics)
            self._default_instances()

        self._classifier_model = _classifier_model
        self.classifier_instance = pipeline(
            "text-classification",
            model=self._classifier_model,
            tokenizer=self._classifier_model,
            truncation=True,
        )

    def get_stereotype_scores(self, responses: List[str]) -> Dict[str, Any]:
        """
        Calculate stereotype scores for a list of outputs.

        Parameters
        ----------
        responses : list of strings
            A list of generated outputs from a language model on which classifier-based
            stereotype metrics will be calculated.

        Returns
        -------
        dict
            Dictionary containing response-level stereotype scores returned by stereotype classifier
        """
        score_dicts = self.classifier_instance(responses)
        stereotype_scores = {
            key: [d[key] for d in score_dicts] for key in score_dicts[0]
        }

        data = {
            "stereotype_score_" + category.lower(): [] for category in self.categories
        }
        data["response"] = responses
        for score, label in zip(stereotype_scores["score"], stereotype_scores["label"]):
            for category in self.categories:
                data["stereotype_score_" + category.lower()].append(
                    score if label == "stereotype_" + category.lower() else 0.0
                )
        return data

    def evaluate(
        self,
        responses: List[str],
        scores: Optional[List[float]] = None,
        prompts: Optional[List[str]] = None,
        return_data: bool = False,
        categories: List[str] = ["gender", "race"],
        show_progress_bars: bool = True,
        existing_progress_bar: Progress = None,
    ) -> Dict[str, Any]:
        """
        Generate stereotype scores and calculate classifier-based stereotype metrics.

        Parameters
        ----------
        responses : list of strings
            A list of generated output from an LLM.

        scores : list of float, default=None
            A list response-level stereotype score. If None, method will compute it first.

        prompts : list of strings, default=None
            A list of prompts from which `responses` were generated. If provided, metrics should be calculated by prompt
            and averaged across prompts (recommend atleast 25 responses per prompt for Expected maximum and Probability metrics).
            Otherwise, metrics are applied as a single calculation over all responses (only stereotype fraction is calculated).

        return_df : bool, default=True
            Specifies whether to include a dictionary containing response-level stereotype scores in returned result.

        show_progress_bars : bool, default=True
            If True, displays progress bars while evaluating metrics.

        existing_progress_bar : rich.progress.Progress, default=None
            If provided, the progress bar will be updated with the existing progress bar.

        Returns
        -------
        dict
            Dictionary containing two keys: 'metrics', containing all metric values, and 'data', containing response-level stereotype scores.
        """
        if show_progress_bars:
            if existing_progress_bar:
                self.progress_bar = existing_progress_bar
            else:
                completion_text = "[progress.percentage]{task.completed}/{task.total}"
                self.progress_bar = Progress(
                    ConditionalSpinnerColumn(),
                    ConditionalTextColumn("[progress.description]{task.description}"),
                    ConditionalBarColumn(),
                    ConditionalTextPercentageColumn(completion_text),
                    ConditionalTimeElapsedColumn(),
                )
                self.progress_bar.start()
        if categories is not None:
            self.categories = categories

        if not scores:
            evaluation_data = self.get_stereotype_scores(responses)

        result = {}
        for category in self.categories:
            if prompts:
                assert prompts is not None, """
                    Prompts must be provided with corresponding responses for to evaluate toxicity metrics by prompt.
                """
                evaluation_data["prompt"] = prompts
                for metric in self.metrics:
                    result[metric.name + " - " + category] = metric.evaluate(
                        data=evaluation_data,
                        threshold=self.threshold,
                        score_column="stereotype_score_" + category.lower(),
                        show_progress_bars=show_progress_bars,
                        existing_progress_bar=self.progress_bar,
                    )
            else:
                if show_progress_bars and self.progress_bar:
                    self.progress_bar.add_task(
                        "[No Progress Bar] -  Calculating Stereotype Fraction - "
                        + category
                        + "..."
                    )
                elif not existing_progress_bar:
                    print("Calculating Stereotype Fraction - " + category + "...")
                result["Stereotype Fraction - " + category] = (
                    Fraction().metric_function(
                        evaluation_data["stereotype_score_" + category.lower()],
                        self.threshold,
                    )
                )

        time.sleep(0.1)
        if self.progress_bar and not existing_progress_bar:
            self.progress_bar.add_task(
                "[No Progress Bar] -  Evaluated metrics successfully!"
            )
            self.progress_bar.stop()
            self.progress_bar = None
        elif not existing_progress_bar:
            print("Evaluated metrics successfully!")
        # If specified, return dataframe
        if return_data:
            return {"metrics": result, "data": evaluation_data}
        return {"metrics": result}

    def _default_instances(self) -> None:
        """Defines default instances of stereotype classifier metrics."""
        self.metrics = []
        for name in self.metric_names:
            tmp = DefaultMetricObjects[name]
            tmp.name = name
            self.metrics.append(tmp)

    def _validate_metrics(self, metric_names: List[str]) -> None:
        """Validate that specified metrics are supported."""
        for name in metric_names:
            assert name in DefaultMetricNames, (
                """Provided metric name is not part of available metrics."""
            )

    def _validate_categories(self, categories: List[str]) -> None:
        """Validate that specified categories are supported."""
        for category in categories:
            assert category.lower() in AvailableCategories, (
                """Provided category name is not part of supported categories."""
            )
