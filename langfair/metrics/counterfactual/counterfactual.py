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

from typing import Any, Dict, Union

import numpy as np

from langfair.generator.counterfactual import CounterfactualGenerator
from langfair.metrics.counterfactual import metrics
from langfair.metrics.counterfactual.metrics.baseclass.metrics import Metric

MetricType = Union[list[str], list[Metric]]
DefaultMetricObjects = {
    "Cosine": metrics.CosineSimilarity(transformer="all-MiniLM-L6-v2", how="pairwise"),
    "Rougel": metrics.RougelSimilarity(how="pairwise"),
    "Bleu": metrics.BleuSimilarity(how="pairwise"),
    "Sentiment Bias": metrics.SentimentBias(how="pairwise"),
}
DefaultMetricNames = list(DefaultMetricObjects.keys())


################################################################################
# Calculate Counterfactual Metrics
################################################################################
class CounterfactualMetrics:
    def __init__(
        self, metrics: MetricType = DefaultMetricNames, neutralize_tokens: str = True
    ) -> None:
        """
        This class computes few or all counterfactual metrics supported LangFair. For more information on these metrics,
        see Huang et al. (2020) :footcite:`huang2020reducingsentimentbiaslanguage` and Bouchard (2024) :footcite:`bouchard2024actionableframeworkassessingbias`.

        Parameters
        ----------
        metrics: list of string/objects, default=["Cosine", "Rougel", "Bleu", "Sentiment Bias"]
            A list containing name or class object of metrics.

        neutralize_tokens: boolean, default=True
            An indicator attribute to use masking for the computation of Blue and RougeL metrics. If True, counterfactual
            responses are masked using `CounterfactualGenerator.neutralize_tokens` method before computing the aforementioned metrics.
        """
        self.metrics = metrics
        if isinstance(metrics[0], str):
            self.metric_names = metrics
            self._validate_metrics(metrics)
            self._default_instances()
        self.neutralize_tokens = neutralize_tokens
        if self.neutralize_tokens:
            self.cf_generator = CounterfactualGenerator()

    def evaluate(
        self,
        texts1: list,
        texts2: list,
        attribute: str = None,
        return_data: bool = False,
    ) -> Dict[str, Any]:
        """
        This method evaluate the counterfactual metrics values for the provided pair of texts.

        Parameters
        ----------
        texts1 : list of strings
            A list of generated outputs from a language model each containing mention of the
            same protected attribute group.

        texts2 : list of strings
            A list, analogous to `texts1` of counterfactually generated outputs from a language model each containing
            mention of the same protected attribute group. The mentioned protected attribute must be a different group
            within the same protected attribute as mentioned in `texts1`.

        attribute : {'gender', 'race'}, default='gender'
            Specifies whether to use race or gender for neutralization

        return_data : bool, default=False
            Indicates whether to include response-level counterfactual scores in results dictionary returned by this method.

        Returns
        -------
        dict
            Dictionary containing values of counterfactual metrics

        References
        ----------
        .. footbibliography::
        """
        if self.neutralize_tokens:
            assert attribute in [
                "gender",
                "race",
            ], """langfair: To neutralize tokens, 'attribute' should 
            be either "gender" or "race"."""
            masked_texts1 = self.cf_generator.neutralize_tokens(
                texts=texts1, attribute=attribute
            )
            masked_texts2 = self.cf_generator.neutralize_tokens(
                texts=texts2, attribute=attribute
            )
        metric_values = {}
        response_scores = {"texts1": texts1, "texts2": texts2}
        for metric in self.metrics:
            if (
                metric.name in ["Bleu Similarity", "RougeL Similarity"]
                and self.neutralize_tokens
            ):
                scores = metric.evaluate(texts1=masked_texts1, texts2=masked_texts2)
            else:
                scores = metric.evaluate(texts1=texts1, texts2=texts2)
            response_scores[metric.name] = scores

            if metric.name == "Sentiment Bias":
                metric_values[metric.name] = metric.parity_value
            else:
                metric_values[metric.name] = np.mean(scores)

        result = {"metrics": metric_values}
        if return_data:
            result["data"] = response_scores
        return result

    def _default_instances(self):
        """Define default metrics."""
        self.metrics = []
        for name in self.metric_names:
            self.metrics.append(DefaultMetricObjects[name])

    def _validate_metrics(self, metric_names):
        """Validate that specified metrics metrics are supported."""
        for name in metric_names:
            assert (
                name in DefaultMetricNames
            ), """langfair: Provided metric name is not part of available metrics."""
