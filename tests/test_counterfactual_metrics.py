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

import json
import os
import platform
import unittest

import pytest

from langfair.metrics.counterfactual import CounterfactualMetrics
from langfair.metrics.counterfactual.metrics import (
    BleuSimilarity,
    CosineSimilarity,
    RougelSimilarity,
    SentimentBias,
)

datafile_path = "tests/data/counterfactual/counterfactual_data_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

actual_result_file_path = "tests/data/counterfactual/counterfactual_results_file.json"
with open(actual_result_file_path, "r") as f:
    actual_results = json.load(f)


def test_bleu():
    bleu = BleuSimilarity()
    x = bleu.evaluate(data["text1"], data["text2"])
    assert x == pytest.approx(actual_results["test1"], rel=1e-02)


@unittest.skipIf(
    ((os.getenv("CI") == "true") & (platform.system() == "Darwin")),
    "Skipping test in macOS CI due to memory issues.",
)
def test_cosine(monkeypatch):
    MOCKED_EMBEDDINGS = actual_results["embeddings"]

    def mock_get_embeddings(*args, **kwargs):
        return MOCKED_EMBEDDINGS

    cosine = CosineSimilarity(transformer="all-MiniLM-L6-v2")
    monkeypatch.setattr(cosine, "_get_embeddings", mock_get_embeddings)
    x = cosine.evaluate(data["text1"], data["text2"])
    assert x == pytest.approx(actual_results["test2"], rel=1e-02)


def test_rougel():
    rougel = RougelSimilarity()
    assert rougel.evaluate(data["text1"], data["text2"]) == actual_results["test3"]


def test_sentiment1():
    sentiment = SentimentBias()
    assert sentiment.evaluate(data["text1"], data["text2"]) == actual_results["test4"]


def test_sentiment2():
    sentiment = SentimentBias(parity="weak")
    assert sentiment.evaluate(data["text1"], data["text2"]) == pytest.approx(
        actual_results["test5"], rel=1e-02
    )


def test_sentiment3(monkeypatch):
    group1 = actual_results["classifier_result1"]
    group2 = actual_results["classifier_result2"]

    def mock_get_classifier(texts, return_all_scores=True):
        if texts in [[t] for t in data["text1"]]:
            idx = data["text1"].index(texts[0])
            return [group1[idx]]
        elif texts in [[t] for t in data["text2"]]:
            idx = data["text2"].index(texts[0])
            return [group2[idx]]
        else:
            return [[]]

    sentiment = SentimentBias(classifier="roberta")
    monkeypatch.setattr(sentiment, "classifier_instance", mock_get_classifier)
    assert sentiment.evaluate(data["text1"], data["text2"]) == pytest.approx(
        actual_results["test7"], rel=1e-02
    )


def test_CounterfactualMetrics():
    metrics = [
        "Rougel",
        "Bleu",
        "Sentiment Bias",
    ]
    counterfactualmetrics = CounterfactualMetrics(metrics=metrics)
    result = counterfactualmetrics.evaluate(
        data["text1"], data["text2"], attribute="race"
    )
    score = result["metrics"]
    ans = actual_results["test6"]["metrics"]
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)


def test_CounterfactualMetrics_VaderClassifier():
    metrics = [
        "Rougel",
        "Bleu",
        "Sentiment Bias",
    ]
    counterfactualmetrics = CounterfactualMetrics(
        metrics=metrics, sentiment_classifier="vader"
    )
    result = counterfactualmetrics.evaluate(
        data["text1"], data["text2"], attribute="race"
    )
    score = result["metrics"]
    ans = actual_results["test6"]["metrics"]
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)
