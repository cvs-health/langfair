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

import json
import os
import platform
import unittest

import pytest

from langfair.auto import SelfEval

datafile_path = "tests/data/autoeval/autoeval_results_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

unique_prompts = list(dict.fromkeys(data["prompts"]))
male_responses = data["counterfactual_responses"]["gender"]["data"]["male_response"]
female_responses = data["counterfactual_responses"]["gender"]["data"]["female_response"]


def test_selfeval_prompt_list():
    """
    Verify the `get_prompts()` contract: originals first, then one segment per
    gender group, each prompt repeated `count` times consecutively, with
    `prompt_manifest` reporting matching segment boundaries.
    """
    se = SelfEval(prompts=unique_prompts, count=25)
    flat = se.get_prompts()

    # Originals first, expanded prompt-major (identical to what AutoEval generates)
    assert len(flat) == 275
    assert flat[0:125] == data["prompts"]

    # One counterfactual segment per gender group, no race segments
    manifest = se.prompt_manifest
    assert [(seg["kind"], seg["attribute"], seg["group"]) for seg in manifest] == [
        ("original", None, None),
        ("counterfactual", "gender", "male"),
        ("counterfactual", "gender", "female"),
    ]
    assert [(seg["start"], seg["end"]) for seg in manifest] == [
        (0, 125),
        (125, 200),
        (200, 275),
    ]
    assert all(seg["count"] == 25 for seg in manifest)
    assert all(
        seg["n_unique_prompts"] == 3 for seg in manifest if seg["kind"] != "original"
    )

    # Each prompt appears `count` times consecutively within each segment
    for start, end in [(0, 125), (125, 200), (200, 275)]:
        for block_start in range(start, end, 25):
            assert len(set(flat[block_start : block_start + 25])) == 1

    # Counterfactual variants differ between groups
    assert flat[125:200] != flat[200:275]


def test_selfeval_validation():
    """
    Verify input validation: response lists of the wrong length or type raise
    errors naming the offending segment, invalid `count` and mismatched
    prompt/response lengths are rejected at construction, and neutral prompts
    with provided responses leave nothing to generate.
    """
    se = SelfEval(prompts=unique_prompts, count=25)

    with pytest.raises(
        ValueError, match="expected 275 responses but received 5"
    ) as excinfo:
        se.evaluate(responses=["a"] * 5)
    assert "counterfactual gender/male" in str(excinfo.value)

    with pytest.raises(TypeError, match="must be strings"):
        se.evaluate(responses=["a"] * 274 + [42])

    with pytest.raises(ValueError, match="positive integer"):
        SelfEval(prompts=unique_prompts, count=0)

    with pytest.raises(ValueError, match="same length"):
        SelfEval(prompts=unique_prompts, responses=["a"] * 3)

    # Neutral prompts with provided responses require no further generation
    se_neutral = SelfEval(
        prompts=["What is the capital of France?"], responses=["Paris."], count=25
    )
    assert se_neutral.get_prompts() == []


def test_selfeval_no_counterfactual_metric():
    """
    Verify that excluding "counterfactual" from `metrics` omits the
    counterfactual segments, so `get_prompts()` returns only the originals.
    """
    se = SelfEval(prompts=unique_prompts, count=25, metrics=["toxicity", "stereotype"])
    assert [seg["kind"] for seg in se.prompt_manifest] == ["original"]
    assert len(se.get_prompts()) == 125


@unittest.skipIf(
    ((os.getenv("CI") == "true") & (platform.system() == "Darwin")),
    "Skipping test in macOS CI due to memory issues.",
)
def test_selfeval_metrics():
    """
    End-to-end check of the two-phase workflow: feed the AutoEval fixture
    responses into `evaluate()` and verify toxicity, stereotype, and
    counterfactual metric values match the fixture results (parity with
    AutoEval), and that `export_results()` writes a results file.
    """
    se = SelfEval(
        prompts=unique_prompts,
        count=25,
        metrics={
            "counterfactual": ["Rougel", "Bleu", "Sentiment Bias"],
            "stereotype": ["Stereotype Association", "Cooccurrence Bias"],
            "toxicity": [
                "Toxic Fraction",
                "Expected Maximum Toxicity",
                "Toxicity Probability",
            ],
        },
    )
    responses = data["responses"] + male_responses + female_responses
    results = se.evaluate(responses=responses, return_data=True)

    file_exist = False
    se.export_results()
    if os.path.exists("results.txt"):
        file_exist = True
        os.remove("results.txt")

    assert file_exist == True  # noqa: E712
    score, ans = results["metrics"]["Toxicity"], data["toxicity_metrics"]
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)
    score, ans = results["metrics"]["Stereotype"], data["stereotype_metrics"]
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)
    score, ans = (
        results["metrics"]["Counterfactual"]["male-female"],
        data["counterfactual_metrics"],
    )
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)


@unittest.skipIf(
    ((os.getenv("CI") == "true") & (platform.system() == "Darwin")),
    "Skipping test in macOS CI due to memory issues.",
)
def test_selfeval_preexisting_responses():
    """
    Verify the pre-existing responses path: with `responses` supplied at
    construction, `get_prompts()` returns only the counterfactual variants,
    and evaluating with those still reproduces the fixture metric values.
    """
    se = SelfEval(prompts=data["prompts"], responses=data["responses"], count=1)

    # Only the counterfactual prompt variants remain to be generated
    flat = se.get_prompts()
    assert len(flat) == 150
    assert [(seg["kind"], seg["group"]) for seg in se.prompt_manifest] == [
        ("counterfactual", "male"),
        ("counterfactual", "female"),
    ]

    results = se.evaluate(responses=male_responses + female_responses, return_data=True)

    score, ans = results["metrics"]["Toxicity"], data["toxicity_metrics"]
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)
    score, ans = results["metrics"]["Stereotype"], data["stereotype_metrics"]
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)
    score, ans = (
        results["metrics"]["Counterfactual"]["male-female"],
        data["counterfactual_metrics"],
    )
    for key in ans:
        assert score[key] == pytest.approx(ans[key], abs=1e-02)
