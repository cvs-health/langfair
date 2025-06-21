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
from math import isclose

import torch

from langfair.metrics.toxicity import AvailableClassifiers, ToxicityMetrics


class TestToxicityMetrics(unittest.TestCase):
    def test_evaluate(self):
        toxic_responses_file = "toxic_responses.csv"
        data_folder = "tests/data/toxicity"
        with open(f"{data_folder}/{toxic_responses_file}", "r") as f:
            toxic = json.load(f)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Use GPU if available
        for classifier in AvailableClassifiers:
            if (
                (classifier in ["roberta-hate-speech-dynabench-r4-target", "toxigen"])
                and (os.getenv("CI") == "true")
                and (platform.system() == "Darwin")
            ):
                continue  # skips CI unit test in macos to avoid memory error

            detoxify = ToxicityMetrics(
                classifiers=[classifier],
                batch_size=100,
                device=device,
                toxic_threshold=0.01,
            )  # TODO: download and pass the model
            toxic_results = detoxify.evaluate(
                responses=toxic.get("data").get("response"),
                prompts=toxic.get("data").get("prompt"),
            )

            with open(f"{data_folder}/toxic_results_{classifier}.json", "r") as f:
                toxic_results_actual = json.load(f)
                print(f"Actual result:{toxic_results_actual}")

            assert isclose(
                toxic_results.get("metrics").get("Toxic Fraction"),
                toxic_results_actual.get("metrics").get("Toxic Fraction"),
            )

            assert isclose(
                toxic_results.get("metrics").get("Expected Maximum Toxicity"),
                toxic_results_actual.get("metrics").get("Expected Maximum Toxicity"),
                abs_tol=1e-06,
            )

            assert isclose(
                toxic_results.get("metrics").get("Toxicity Probability"),
                toxic_results_actual.get("metrics").get("Toxicity Probability"),
            )


if __name__ == "__main__":
    unittest.main()
