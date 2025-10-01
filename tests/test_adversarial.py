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

import gc
import glob
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from langfair.generator.redteaming import (
    INSTRUCTION_DICT,
    AdversarialGenerator,
)


@pytest.fixture(scope="session", autouse=True)
def cleanup_disk():
    """Clean up temp and cache directories to free disk space in CI."""
    try:
        temp_dir = tempfile.gettempdir()
        for path in glob.glob(os.path.join(temp_dir, "*")):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)

        shutil.rmtree(os.path.expanduser("~/.cache"), ignore_errors=True)
        print("Disk cleanup completed.")
    except Exception as e:
        print("Disk cleanup failed:", e)


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Force cleanup after each test"""
    yield
    gc.collect()


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.temperature = 0.7
    return llm


@pytest.fixture
def generator(mock_llm):
    gen = AdversarialGenerator(langchain_llm=mock_llm)
    gen.count = 1
    return gen


@pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.platform.startswith("linux"),
    reason="Skip on Ubuntu CI due to disk space",
)
@pytest.mark.asyncio
async def test_generate_from_template_valid(generator):
    prompts = {"text": ["Alice is a", "Bob is a"]}
    generator.generate_responses = AsyncMock(
        return_value={"data": {"response": ["Alice is a teacher", "Bob is a doctor"]}}
    )
    result = await generator._generate_from_template(prompts, ["benign"], count=1)
    assert "benign_response" in result
    assert len(result["benign_response"]) == 2


@pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.platform.startswith("linux"),
    reason="Skip on Ubuntu CI due to disk space",
)
@pytest.mark.asyncio
async def test_generate_from_template_invalid_style(generator):
    with pytest.raises(ValueError):
        await generator._generate_from_template(
            {"text": ["test"]}, ["invalid"], count=1
        )


@pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.platform.startswith("linux"),
    reason="Skip on Ubuntu CI due to disk space",
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
    assert result["data"]["prompt"] == ["X is Y", "Z is W"]
    assert result["metadata"]["benign_response_non_completion_rate"] == 0.5
    assert result["metadata"]["adversarial_response_non_completion_rate"] == 1.0


@pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.platform.startswith("linux"),
    reason="Skip on Ubuntu CI due to disk space",
)
def test_read_counterfactual_data_valid():
    mock_json = json.dumps(
        [
            {
                "text": "X is Y",
                "group": "Homosexuals",
                "group_category": "Sexual orientation",
            }
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


@pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.platform.startswith("linux"),
    reason="Skip on Ubuntu CI due to disk space",
)
def test_read_counterfactual_data_invalid():
    with pytest.raises(ValueError):
        AdversarialGenerator._read_counterfactual_data(["InvalidGroup"])


@pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.platform.startswith("linux"),
    reason="Skip on Ubuntu CI due to disk space",
)
def test_read_toxicity_data_toxic():
    mock_lines = [
        json.dumps({"prompt": {"text": "test", "toxicity": 0.9}, "challenging": True})
    ]
    with (
        patch("builtins.open", mock_open(read_data="\n".join(mock_lines))),
        patch(
            "pkgutil.resolve_name",
            return_value=MagicMock(__file__=os.path.join("langfair", "__init__.py")),
        ),
        patch("random.sample", return_value=["test"]),
    ):
        result = AdversarialGenerator._read_toxicity_data("toxic", "benign", 1, 0.1)
        assert result[0].startswith(INSTRUCTION_DICT["benign"])


@pytest.mark.skipif(
    os.getenv("CI") == "true" and sys.platform.startswith("linux"),
    reason="Skip on Ubuntu CI due to disk space",
)
def test_read_toxicity_data_nontoxic():
    mock_lines = [
        json.dumps({"prompt": {"text": "test", "toxicity": 0.05}, "challenging": False})
    ]
    with (
        patch("builtins.open", mock_open(read_data="\n".join(mock_lines))),
        patch(
            "pkgutil.resolve_name",
            return_value=MagicMock(__file__=os.path.join("langfair", "__init__.py")),
        ),
        patch("random.sample", return_value=["test"]),
    ):
        result = AdversarialGenerator._read_toxicity_data("nontoxic", "benign", 1, 0.1)
        assert result[0].startswith(INSTRUCTION_DICT["benign"])
