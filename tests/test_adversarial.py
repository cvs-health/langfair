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
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from langfair.generator.redteaming import (
    INSTRUCTION_DICT,
    AdversarialGenerator,
)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.temperature = 0.7
    return llm


@pytest.fixture
def generator(mock_llm):
    gen = AdversarialGenerator(langchain_llm=mock_llm)
    gen.count = 2  # override default for test simplicity
    return gen


@pytest.mark.asyncio
async def test_generate_from_template_valid(generator):
    prompts = {"text": ["Alice is a", "Bob is a"]}
    generator.generate_responses = AsyncMock(
        return_value={"data": {"response": ["Alice is a teacher", "Bob is a doctor"]}}
    )
    result = await generator._generate_from_template(prompts, ["benign"], count=1)
    assert "benign_response" in result
    assert len(result["benign_response"]) == 2


@pytest.mark.asyncio
async def test_generate_from_template_invalid_style(generator):
    with pytest.raises(ValueError):
        await generator._generate_from_template(
            {"text": ["test"]}, ["invalid"], count=1
        )


def test_format_result_structure(generator):
    dataset = {
        "benign_response": ["ok", "Unable to get response"],
        "adversarial_response": ["Unable to get response", "Unable to get response"],
    }
    prompt_templates = {
        "text": ["X is Y", "Z is W"],
        "group": ["A", "B"],
        "group_category": ["Race", "Gender"],
    }
    result = generator._format_result(
        dataset, prompt_templates, keys=["text", "group", "group_category"]
    )
    assert "data" in result
    assert "metadata" in result
    assert result["data"]["prompt"] == ["X is Y", "X is Y", "Z is W", "Z is W"]
    assert result["metadata"]["benign_response_non_completion_rate"] == 0.5
    assert result["metadata"]["adversarial_response_non_completion_rate"] == 1.0


def test_read_counterfactual_data_valid():
    mock_json = json.dumps(
        [
            {
                "text": "X is Y",
                "group": "Homosexuals",
                "group_category": "Sexual orientation",
            },
            {
                "text": "Z is W",
                "group": "Able-bodied people",
                "group_category": "Disability",
            },
        ]
    )
    with (
        patch("builtins.open", mock_open(read_data=mock_json)),
        patch(
            "pkgutil.resolve_name",
            return_value=MagicMock(__file__=os.path.join("langfair", "__init__.py")),
        ),
    ):
        result = AdversarialGenerator._read_counterfactual_data(["Sexual orientation"])
        assert "text" in result
        assert result["group"] == ["Homosexuals"]
        assert result["group_category"] == ["Sexual orientation"]


def test_read_counterfactual_data_invalid():
    with pytest.raises(ValueError):
        AdversarialGenerator._read_counterfactual_data(["InvalidGroup"])


def test_read_toxicity_data_toxic():
    mock_lines = [
        json.dumps(
            {
                "prompt": {"text": "Why are cats evil?", "toxicity": 0.9},
                "challenging": True,
            }
        ),
        json.dumps(
            {
                "prompt": {"text": "Why do people lie?", "toxicity": 0.8},
                "challenging": False,
            }
        ),
    ]
    with (
        patch("builtins.open", mock_open(read_data="\n".join(mock_lines))),
        patch(
            "pkgutil.resolve_name",
            return_value=MagicMock(__file__=os.path.join("langfair", "__init__.py")),
        ),
        patch("random.sample", return_value=["Why are cats evil?"]),
    ):
        result = AdversarialGenerator._read_toxicity_data("toxic", "benign", 1, 0.1)
        assert result[0].startswith(INSTRUCTION_DICT["benign"])


def test_read_toxicity_data_nontoxic():
    mock_lines = [
        json.dumps(
            {
                "prompt": {"text": "Why are cats cute?", "toxicity": 0.05},
                "challenging": False,
            }
        ),
        json.dumps(
            {
                "prompt": {"text": "Why do people smile?", "toxicity": 0.2},
                "challenging": False,
            }
        ),
    ]
    with (
        patch("builtins.open", mock_open(read_data="\n".join(mock_lines))),
        patch(
            "pkgutil.resolve_name",
            return_value=MagicMock(__file__=os.path.join("langfair", "__init__.py")),
        ),
        patch("random.sample", return_value=["Why are cats cute?"]),
    ):
        result = AdversarialGenerator._read_toxicity_data("nontoxic", "benign", 1, 0.1)
        assert result[0].startswith(INSTRUCTION_DICT["benign"])
