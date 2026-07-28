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

import numpy as np
import pytest

from langfair.metrics.stereotype import StereotypeMetrics
from langfair.metrics.stereotype.metrics import (
    CooccurrenceBiasMetric,
    StereotypeClassifier,
    StereotypicalAssociations,
)

datafile_path = "tests/data/stereotype/stereotype_data_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

actual_result_file_path = "tests/data/stereotype/stereotype_results_file.json"
with open(actual_result_file_path, "r") as f:
    actual_results = json.load(f)


def test_associations1():
    association = StereotypicalAssociations(target_category="adjective")
    x = association.evaluate(responses=data["responses"])
    assert x == actual_results["test1"]


def test_associations2():
    association = StereotypicalAssociations(target_category="profession")
    x = association.evaluate(responses=data["responses_profession"])
    assert x == actual_results["test2"]


@pytest.mark.parametrize(
    "first, second",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_associations_reuse_across_progress_bar_settings(first, second):
    """A single instance must be reusable with `show_progress_bars` in any order.

    Previously `progress_bar_task` was assigned only under `show_progress_bars`
    but read under `self.progress_bar`, which stayed truthy after the first
    call with progress bars on, raising UnboundLocalError on the next call
    with them off.
    """
    association = StereotypicalAssociations(target_category="adjective")
    expected = actual_results["test1"]

    assert association.evaluate(data["responses"], show_progress_bars=first) == expected
    assert (
        association.evaluate(data["responses"], show_progress_bars=second) == expected
    )


def test_associations_returns_none_without_cooccurrences(recwarn):
    """Responses with no target/demographic co-occurrence yield None, not nan.

    The `if not bias_scores` guard used to run *after* `np.array([]).mean()`,
    emitting "Mean of empty slice" and dividing by zero before returning.
    """
    association = StereotypicalAssociations(target_category="adjective")
    neutral = ["The weather is fine today.", "Numbers go up and down."]

    assert association.evaluate(neutral, show_progress_bars=False) is None
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_associations_no_empty_mean_under_numpy_raise():
    """The degenerate case must not trip pipelines that set np.seterr(all='raise')."""
    association = StereotypicalAssociations(target_category="adjective")
    neutral = ["The weather is fine today.", "Numbers go up and down."]

    old = np.seterr(all="raise")
    try:
        assert association.evaluate(neutral, show_progress_bars=False) is None
    finally:
        np.seterr(**old)


def test_coocurrence1():
    cobs = CooccurrenceBiasMetric(target_category="adjective")
    x = cobs.evaluate(responses=data["responses"])
    assert x == pytest.approx(actual_results["test3"], rel=1e-02)


def test_coocurrence2():
    cobs = CooccurrenceBiasMetric(target_category="profession")
    x = cobs.evaluate(responses=data["responses_profession"])
    assert x == pytest.approx(actual_results["test4"], rel=1e-02)


@unittest.skipIf(
    ((os.getenv("CI") == "true") & (platform.system() == "Darwin")),
    "Skipping test in macOS CI due to memory issues.",
)
def test_classifier1():
    classifier = StereotypeClassifier(metrics=["Stereotype Fraction"])
    x = classifier.evaluate(responses=data["responses_fraction"], return_data=True)
    assert x["metrics"] == actual_results["test5"]["metrics"]
    assert x["data"]["response"] == data["responses_fraction"]


@unittest.skipIf(
    ((os.getenv("CI") == "true") & (platform.system() == "Darwin")),
    "Skipping test in macOS CI due to memory issues.",
)
def test_classifier2():
    classifier = StereotypeClassifier()
    score = classifier.evaluate(
        responses=data["responses_fraction"], prompts=data["prompts"], return_data=False
    )
    ans = actual_results["test6"]["metrics"]
    for key in ans:
        assert score["metrics"][key] == pytest.approx(ans[key], abs=1e-02)


@unittest.skipIf(
    ((os.getenv("CI") == "true") & (platform.system() == "Darwin")),
    "Skipping test in macOS CI due to memory issues.",
)
def test_StereotypeMetrics():
    stereotypemetrics = StereotypeMetrics()
    score = stereotypemetrics.evaluate(
        responses=data["responses_fraction"], prompts=data["prompts"]
    )
    ans = actual_results["test7"]["metrics"]
    for key in ans:
        assert score["metrics"][key] == pytest.approx(ans[key], abs=1e-02)
