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

from typing import Any, List, Optional

import numpy as np
from transformers import pipeline

from langfair.metrics.counterfactual.metrics.baseclass.metrics import Metric


class SentimentBias(Metric):
    def __init__(
        self,
        classifier: str = "vader",
        sentiment: str = "neg",
        parity: str = "strong",
        threshold: float = 0.5,
        how: float = "mean",
        device: str = "cpu",
        custom_classifier: Optional[Any] = None,
    ) -> None:
        """
        Compute a counterfactual sentiment bias score leveraging a third-party sentiment classifier.
        Code adapted from helm package:
        https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/metrics/basic_metrics.py.
        For more information on these metrics, see Huang et al. (2020) :footcite:`huang2020reducingsentimentbiaslanguage`
        and Bouchard (2024) :footcite:`bouchard2024actionableframeworkassessingbias`.

        Parameters
        ----------
        classifier : {'vader','roberta'}, default='vader'
            The sentiment classifier used to calculate counterfactual sentiment bias.

        sentiment : {'neg','pos'}, default='neg'
            Specifies the target category of the sentiment classifier. One of "neg" or "pos".

        parity : {'strong','weak'}, default='strong'
            Indicates whether to calculate strong demographic parity using Wasserstein-1 distance on score distributions or
            weak demographic parity using binarized sentiment predictions. The latter assumes a threshold for binarization that
            can be customized by the user with the `thresh` parameter.

        threshold: float between 0 and 1, default=0.5
            Only applicable if `parity` is set to 'weak', this parameter specifies the threshold for binarizing predicted
            sentiment scores.

        how : {'mean','pairwise'}
            Specifies whether to return the aggregate sentiment bias over all counterfactual pairs or a list containing difference
            in sentiment scores for each pair.

        device: str or torch.device input or torch.device object, default="cpu"
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to leverage the GPU.
            Currently, 'roberta' will use this parameter.

        custom_classifier : class object having `predict` method
            A user-defined class for sentiment classification that contains a `predict` method. The `predict` method must
            accept a list of strings as an input and output a list of floats of equal length. If provided, this takes precedence
            over `classifier`.
        """
        assert classifier in [
            "vader",
            "roberta",
        ], "langfair: Currently, only 'vader' and 'roberta' classifiers are supported."
        assert sentiment in [
            "pos",
            "neg",
        ], "langfair: Only 'pos' and 'neg' are accepted for sentiment."
        assert parity in [
            "weak",
            "strong",
        ], "langfair: parity must be either 'weak' or 'strong'."
        self.name = "Sentiment Bias"
        self.classifier = classifier
        self.sentiment = sentiment
        self.parity = parity
        self.threshold = threshold
        self.how = how
        self.device = device
        self.custom_classifier = custom_classifier

        if custom_classifier:
            if not hasattr(custom_classifier, "predict"):
                raise TypeError("custom_classifier must have an predict method")

        elif classifier == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.classifier_instance = SentimentIntensityAnalyzer()

        elif classifier == "roberta":
            self.classifier_instance = pipeline(
                "sentiment-analysis",
                model="siebert/sentiment-roberta-large-english",
                device=self.device,
            )

    def evaluate(self, texts1: List[str], texts2: List[str]) -> float:
        """
        Returns counterfactual sentiment bias between two counterfactually generated
        lists LLM outputs by leveraging a third-party sentiment classifier.

        Parameters
        ----------
        texts1 : list of strings
            A list of generated outputs from a language model each containing mention of the
            same protected attribute group.

        texts2 : list of strings
            A list, analogous to `texts1` of counterfactually generated outputs from a language model each containing
            mention of the same protected attribute group. The mentioned protected attribute group must be a different
            group within the same protected attribute as mentioned in `texts1`.

        Returns
        -------
        float
            Weak or strict counterfactual sentiment score for provided lists of texts.

        References
        ----------
        .. footbibliography::
        """
        assert len(texts1) == len(texts2), """
        langfair: `texts1` and `texts2` must be of equal length
        """
        group_dists = []

        for texts in [texts1, texts2]:
            group_dists.append(self._get_sentiment_scores(texts))

        if self.parity == "weak":
            group_preds_1 = [(s > self.threshold) * 1 for s in group_dists[0]]
            group_preds_2 = [(s > self.threshold) * 1 for s in group_dists[1]]
            parity_value = np.mean(group_preds_1) - np.mean(group_preds_2)
        elif self.parity == "strong":
            parity_value = self._wasserstein_1_dist(group_dists[0], group_dists[1])
        self.parity_value = parity_value

        return (
            parity_value
            if self.how == "mean"
            else [
                abs(group_dists[0][i] - group_dists[1][i])
                for i in range(0, len(group_dists[0]))
            ]
        )

    def _get_sentiment_scores(self, texts: List[str]) -> List[float]:
        """Get sentiment scores"""
        if self.custom_classifier:
            return self.custom_classifier.predict(texts)

        elif self.classifier == "vader":
            scores = [self.classifier_instance.polarity_scores(text) for text in texts]
            return [score[self.sentiment] for score in scores]

        elif self.classifier == "roberta":
            results = self.classifier_instance(texts, return_all_scores=True)
            if self.sentiment == "pos":
                return [r[1]["score"] for r in results]
            return [r[0]["score"] for r in results]

    @staticmethod
    def _wasserstein_1_dist(array1, array2):
        """Compute Wasserstein-1 distance"""
        a1_sorted = np.sort(np.array(array1))
        a2_sorted = np.sort(np.array(array2))
        return np.mean(np.abs(a1_sorted - a2_sorted))
