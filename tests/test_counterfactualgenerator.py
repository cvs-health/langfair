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

import pytest
import warnings
from unittest.mock import Mock, MagicMock
from langchain_openai import AzureChatOpenAI

from langfair.generator import CounterfactualGenerator


@pytest.mark.asyncio
async def test_counterfactual(monkeypatch):
    # TODO: Tests to check if `parse_texts` method works for all words in gender/race word list.
    # TODO: Add tests for `estimate_token_cost` method (first need to fix the bug)
    MOCKED_PROMPTS = [
        "prompt 1: male person",
        "prompt 2: female person",
        "prompt 3: white person",
        "prompt 4: black person",
    ]
    MOCKED_RACE_PROMPTS = {
        "white_prompt": ["prompt 3: white person", "prompt 4: white person"],
        "black_prompt": ["prompt 3: black person", "prompt 4: black person"],
        "hispanic_prompt": ["prompt 3: hispanic person", "prompt 4: hispanic person"],
        "asian_prompt": ["prompt 3: asian person", "prompt 4: asian person"],
        "attribute_words": [["white person"], ["black person"]],
        "original_prompt": ["prompt 3: white person", "prompt 4: black person"],
    }
    MOCKED_GENDER_PROMPTS = {
        "male_prompt": ["prompt 1: male person", "prompt 2: male person"],
        "female_prompt": ["prompt 1: female person", "prompt 2: female person"],
        "attribute_words": [["male"], ["female"]],
        "original_prompt": ["prompt 1: male person", "prompt 2: female person"],
    }
    # MOCKED_CF_PROMPTS = list(MOCKED_RACE_PROMPTS.values()) + list(
    #     MOCKED_GENDER_PROMPTS.values()
    # )
    MOCKED_RESPONSES = [
        "Gender response",
        "Race response",
    ]

    async def mock_async_api_call(prompt, *args, **kwargs):
        if "1" in prompt or "2" in prompt:
            return [MOCKED_RESPONSES[0]]
        elif "3" in prompt or "4" in prompt:
            return [MOCKED_RESPONSES[-1]]

    mock_object = AzureChatOpenAI(
        deployment_name="YOUR-DEPLOYMENT",
        temperature=0,
        api_key="SECRET_API_KEY",
        api_version="2024-05-01-preview",
        azure_endpoint="https://mocked.endpoint.com",
    )

    counterfactual_object = CounterfactualGenerator(langchain_llm=mock_object)

    monkeypatch.setattr(counterfactual_object, "_async_api_call", mock_async_api_call)

    race_prompts = counterfactual_object.parse_texts(
        texts=MOCKED_PROMPTS, attribute="race"
    )
    assert race_prompts == [[], [], ["white person"], ["black person"]]

    gender_prompts = counterfactual_object.parse_texts(
        texts=MOCKED_PROMPTS, attribute="gender"
    )
    assert gender_prompts == [["male"], ["female"], [], []]

    race_prompts = counterfactual_object.create_prompts(
        prompts=MOCKED_PROMPTS, attribute="race"
    )
    assert race_prompts == MOCKED_RACE_PROMPTS

    gender_prompts = counterfactual_object.create_prompts(
        prompts=MOCKED_PROMPTS, attribute="gender"
    )
    assert gender_prompts == MOCKED_GENDER_PROMPTS

    cf_data = await counterfactual_object.generate_responses(
        prompts=MOCKED_PROMPTS, attribute="race", count=1
    )
    assert all(
        [
            cf_data["data"][key] == [MOCKED_RESPONSES[-1]] * 2
            for key in cf_data["data"]
            if "response" in key
        ]
    )

    cf_data = await counterfactual_object.generate_responses(
        prompts=MOCKED_PROMPTS, attribute="gender", count=1
    )
    assert all(
        [
            cf_data["data"][key] == [MOCKED_RESPONSES[0]] * 2
            for key in cf_data["data"]
            if "response" in key
        ]
    )


def test_race_parsing_false_positives():
    """Test that _get_race_subsequences avoids false positives like 'asian' in 'caucasian'."""
    from langfair.generator import CounterfactualGenerator

    cf_gen = CounterfactualGenerator()

    # Original issue: should not find "asian" in "caucasian male"
    result = cf_gen._get_race_subsequences(
        "The patient is a caucasian male diagnosed with ABC."
    )
    result_lower = [word.lower() for word in result]

    # Should not contain false positive "asian"
    assert "asian" not in result_lower, (
        f"False positive 'asian' found in result: {result}"
    )

    # Should contain the actual race term (either standalone or with person descriptor)
    assert any("caucasian" in word.lower() for word in result), (
        f"Expected 'caucasian' not found in result: {result}"
    )

    # Additional edge cases to prevent regressions

    # Should not find "american" in "American" (not a race term in our lists)
    result2 = cf_gen._get_race_subsequences("The American patient was treated.")
    assert len(result2) == 0, (
        f"Should not find race terms in non-racial context: {result2}"
    )

    # Should correctly identify actual race terms
    result3 = cf_gen._get_race_subsequences(
        "The hispanic woman was seen by the doctor."
    )
    assert any("hispanic" in word.lower() for word in result3), (
        f"Expected 'hispanic' not found: {result3}"
    )


def test_llm_retry_logic_basic():
    """Test basic pipe parsing and hallucination prevention."""
    cf_gen = CounterfactualGenerator()
    
    # Test pipe format works
    terms, is_formatted = cf_gen._parse_llm_response("he|she", "He said she was coming")
    assert terms == ["he", "she"]
    assert is_formatted == True
    
    # Test hallucination prevention
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        terms, _ = cf_gen._parse_llm_response("he|nonexistent", "He was walking")
        assert terms == ["he"]  # Should drop nonexistent term


def test_llm_retry_logic_malformed_then_success():
    """Test retry logic when LLM gives poorly formatted response first, then succeeds."""
    cf_gen = CounterfactualGenerator()
    
    # Mock LLM that fails first, succeeds on retry
    mock_llm = Mock()
    first_response = Mock()
    first_response.content = "The gender terms are: male, female"  # Poorly formatted
    second_response = Mock()
    second_response.content = "male|female"  # Well formatted
    mock_llm.invoke.side_effect = [first_response, second_response]
    
    result = cf_gen._detect_terms_with_retry("The male and female patients", mock_llm, "gender", "test prompt")
    
    # Should successfully get terms from LLM after retry (not from static fallback)
    assert set(result) == {"male", "female"}  # Retry mechanism should succeed with LLM
    assert mock_llm.invoke.call_count == 2  # Should call LLM twice (first fails, retry succeeds)
    
    # This verifies the retry worked because we're getting the specific terms
    # from the LLM's successful second response


def test_llm_retry_logic_fallback_to_static():
    """Test fallback to static method when LLM fails twice."""
    cf_gen = CounterfactualGenerator()
    
    # Mock LLM that fails both times with clearly poorly formatted responses
    mock_llm = Mock()
    mock_llm.invoke.side_effect = [
        Mock(content="I found these terms in the text but can't format properly"),
        Mock(content="Still giving you a long explanatory response instead of proper format")
    ]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = cf_gen._detect_terms_with_retry("He said she was leaving", mock_llm, "gender", "test prompt")
    
    # Should fall back to static method and successfully find terms
    assert "he" in result and "she" in result  # Static method should find these gender terms
    assert mock_llm.invoke.call_count == 2  # Should have tried LLM twice before fallback
