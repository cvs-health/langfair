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

from typing import Dict, List, Union

from langfair.metrics.stereotype.metrics import (
    CooccurrenceBiasMetric,
    StereotypeClassifier,
    StereotypicalAssociations,
)
from langfair.metrics.stereotype.metrics.baseclass.metrics import Metric

MetricType = Union[list[str], list[Metric]]
DefaultMetricClasses = {
    "Stereotype Association": StereotypicalAssociations,
    "Cooccurrence Bias": CooccurrenceBiasMetric,
    "Stereotype Classifier": StereotypeClassifier,
}
DefaultMetricNames = list(DefaultMetricClasses.keys())


################################################################################
# Calculate Counterfactual Metrics
################################################################################
class StereotypeMetrics:
    def __init__(
        self,
        metrics: MetricType = DefaultMetricNames,
        _classifier_model: str = "wu981526092/Sentence-Level-Stereotype-Detector",
    ) -> None:
        """
        This class computes few or all Stereotype metrics supported langfair. For more information on these metrics, see Liang et al. (2023) :footcite:`liang2023holisticevaluationlanguagemodels`,
        Bordia & Bowman (2019) :footcite:`bordia2019identifyingreducinggenderbias` and Zekun et al. (2023) :footcite:`zekun2023auditinglargelanguagemodels`.

        Parameters
        ----------
        metrics: list of string/objects, default=["Stereotype Association", "Cooccurrence Bias", "Stereotype Classifier"]
            A list containing name or class object of metrics.

        """
        self.metrics = metrics
        self._classifier_model = _classifier_model
        if isinstance(metrics[0], str):
            self.metric_names = metrics
            self._validate_metrics(metrics)
            self._default_instances()

    def evaluate(
        self,
        responses: List[str],
        prompts: List[str] = None,
        return_data: bool = False,
        categories: List[str] = ["gender", "race"],
    ) -> Dict[str, float]:
        """
        This method evaluate the stereotype metrics values for the provided pair of texts.

        Parameters
        ----------
        responses : list of strings
            A list of generated output from an LLM.

        prompts : list of strings, default=None
            A list of prompts from which `responses` were generated. If provided, metrics should be calculated by prompt
            and averaged across prompts (recommend atleast 25 responses per prompt for Expected maximum and Probability metrics).
            Otherwise, metrics are applied as a single calculation over all responses (only stereotype fraction is calculated).

        return_data : bool, default=False
            Specifies whether to include a dictionary containing response-level stereotype scores in returned result.

        categories: list, subset of ['gender', 'race']
            Specifies attributes for stereotype classifier metrics. Includes both race and gender by default.

        Returns
        -------
        dict
            Dictionary containing two keys: 'metrics', containing all metric values, and 'data', containing response-level stereotype scores.

        References
        ----------
        .. footbibliography::
        """
        metric_values = {}
        for metric in self.metrics:
            if metric.name in ["Stereotype Classifier"]:
                tmp_value = metric.evaluate(
                    responses=responses,
                    prompts=prompts,
                    return_data=return_data,
                    categories=categories,
                )
                metric_values.update(tmp_value["metrics"])
            else:
                metric_values[metric.name] = metric.evaluate(responses=responses)
        if return_data:
            return {"metrics": metric_values, "data": tmp_value["data"]}
        return {"metrics": metric_values}

    def _default_instances(self) -> None:
        self.metrics = []
        for name in self.metric_names:
            if name == "Stereotype Classifier":
                self.metrics.append(
                    DefaultMetricClasses[name](_classifier_model=self._classifier_model)
                )
            else:
                self.metrics.append(DefaultMetricClasses[name]())

    def _validate_metrics(self, metric_names: List[str]) -> None:
        for name in metric_names:
            assert (
                name in DefaultMetricNames
            ), """Provided metric name is not part of available metrics."""
