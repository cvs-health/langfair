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

from rich.live import Live

from langfair.metrics.counterfactual import CounterfactualMetrics
from langfair.metrics.counterfactual.metrics import RougelSimilarity
from langfair.metrics.stereotype import StereotypeMetrics
from langfair.metrics.toxicity import ToxicityMetrics
from langfair.utils.display import start_progress_bar, stop_progress_bar


class _ConstantClassifier:
    """Stub toxicity classifier so tests avoid loading real models."""

    def predict(self, texts):
        return [0.1] * len(texts)


RESPONSES = [
    "He is a doctor and she is a nurse.",
    "The man drove while the woman read a book.",
]
PROMPTS = ["Describe scenario one.", "Describe scenario two."]
TEXTS1 = [
    "He went to the market this morning.",
    "The father cooked dinner for everyone.",
]
TEXTS2 = [
    "She went to the market this morning.",
    "The mother cooked dinner for everyone.",
]


def _run_metric_classes_with(bar):
    """Run one metric class from each family, sharing `bar` via existing_progress_bar."""
    tm = ToxicityMetrics(custom_classifier=_ConstantClassifier())
    tm.evaluate(
        responses=RESPONSES,
        prompts=PROMPTS,
        show_progress_bars=True,
        existing_progress_bar=bar,
    )

    sm = StereotypeMetrics(metrics=["Stereotype Association", "Cooccurrence Bias"])
    sm.evaluate(
        responses=RESPONSES,
        show_progress_bars=True,
        existing_progress_bar=bar,
    )

    cm = CounterfactualMetrics(metrics=["Rougel"], neutralize_tokens=False)
    cm.evaluate(
        texts1=TEXTS1,
        texts2=TEXTS2,
        show_progress_bars=True,
        existing_progress_bar=bar,
    )


def test_borrowed_progress_bar_stays_running():
    """
    Verify that metric classes do not stop a progress bar they received via
    `existing_progress_bar`: the borrowed bar must still be running after each
    child finishes, and only the creator's stop call ends it.
    """
    bar = start_progress_bar()
    try:
        _run_metric_classes_with(bar)
        assert bar.live.is_started
    finally:
        stop_progress_bar(bar)
    assert not bar.live.is_started


def test_owned_progress_bar_still_stopped():
    """
    Verify that a metric class that creates its own progress bar (no
    `existing_progress_bar` passed) still stops it when evaluation completes.
    """
    tm = ToxicityMetrics(custom_classifier=_ConstantClassifier())
    tm.evaluate(responses=RESPONSES, prompts=PROMPTS, show_progress_bars=True)
    assert tm.progress_bar is not None
    assert not tm.progress_bar.live.is_started

    rougel = RougelSimilarity()
    rougel.evaluate(texts1=TEXTS1, texts2=TEXTS2, show_progress_bars=True)
    assert rougel.progress_bar is not None
    assert not rougel.progress_bar.live.is_started


def test_shared_bar_opens_single_live_session(monkeypatch):
    """
    Regression test for the repeated progress-display renders: a flow that
    shares one bar across several metric classes must open exactly one rich
    `Live` session, instead of restarting it after every child stops the bar.
    """
    effective_starts = []
    original_start = Live.start

    def counting_start(self, *args, **kwargs):
        if not self.is_started:
            effective_starts.append(1)
        return original_start(self, *args, **kwargs)

    monkeypatch.setattr(Live, "start", counting_start)

    bar = start_progress_bar()
    try:
        _run_metric_classes_with(bar)
    finally:
        stop_progress_bar(bar)

    assert len(effective_starts) == 1
