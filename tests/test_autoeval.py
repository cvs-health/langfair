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
from langchain_openai import AzureChatOpenAI

from langfair.auto import AutoEval

datafile_path = "tests/data/autoeval/autoeval_results_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)


@unittest.skipIf(
    ((os.getenv("CI") == "true") & (platform.system() == "Darwin")),
    "Skipping test in macOS CI due to memory issues.",
)
@pytest.mark.asyncio
async def test_autoeval(monkeypatch):
    mock_llm_object = AzureChatOpenAI(
        deployment_name="YOUR-DEPLOYMENT",
        temperature=1,
        api_key="SECRET_API_KEY",
        api_version="2024-05-01-preview",
        azure_endpoint="https://mocked.endpoint.com",
    )

    async def mock_cf_generate_responses(prompts, attribute, *args, **kwargs):
        return data["counterfactual_responses"][attribute]

    async def mock_generate_responses(*args, **kwargs):
        return {"data": {"prompt": data["prompts"], "response": data["responses"]}}

    ae = AutoEval(
        prompts=data["prompts"],
        langchain_llm=mock_llm_object,
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

    monkeypatch.setattr(
        ae.generator_object, "generate_responses", mock_generate_responses
    )
    monkeypatch.setattr(
        ae.cf_generator_object, "generate_responses", mock_cf_generate_responses
    )

    results = await ae.evaluate(return_data=True)

    file_exist = False
    ae.export_results()
    if os.path.exists("results.txt"):
        file_exist = True
        os.remove("results.txt")

    assert file_exist == True
    score, ans = results["metrics"]["Toxicity"], data["toxicity_metrics"]
    assert all([abs(score[key] - ans[key]) < 1e-5 for key in ans])
    score, ans = results["metrics"]["Stereotype"], data["stereotype_metrics"]
    assert all([abs(score[key] - ans[key]) < 1e-5 for key in ans])
    score, ans = (
        results["metrics"]["Counterfactual"]["male-female"],
        data["counterfactual_metrics"],
    )
    assert all([abs(score[key] - ans[key]) < 1e-5 for key in ans])
