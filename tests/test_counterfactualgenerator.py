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
from langchain_openai import AzureChatOpenAI

from langfair.generator import CounterfactualGenerator


@pytest.mark.asyncio
async def test_counterfactual(monkeypatch):
    # TODO: Tests to check if `parse_texts` method works for all words in gender/race word list.
    # TODO: Add tests for `estimate_token_cost` method (first need to fix the bug)
    MOCKED_PROMPTS = [
        "prompt 1: male",
        "prompt 2: female",
        "prompt 3: white person",
        "prompt 4: black person",
    ]
    MOCKED_RACE_PROMPTS = {
        "white_prompt": ["prompt 3: white person", "prompt 4: white person"],
        "black_prompt": ["prompt 3: black person", "prompt 4: black person"],
        "hispanic_prompt": ["prompt 3: hispanic person", "prompt 4: hispanic person"],
        "asian_prompt": ["prompt 3: asian person", "prompt 4: asian person"],
        "arabic_prompt": ["prompt 3: arabic person", "prompt 4: arabic person"],
        "attribute_words": [["white person"], ["black person"]],
        "original_prompt": ["prompt 3: white person", "prompt 4: black person"],
    }
    MOCKED_GENDER_PROMPTS = {
        "male_prompt": ["prompt 1: male", "prompt 2: male"],
        "female_prompt": ["prompt 1: female", "prompt 2: female"],
        "nonbinary_prompt": ["prompt 1: individual", "prompt 2: individual"],
        "queer_prompt": ["prompt 1: individual", "prompt 2: individual"],
        "attribute_words": [["male"], ["female"]],
        "original_prompt": ["prompt 1: male", "prompt 2: female"],
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


def test_new_attributes():
    cdg = CounterfactualGenerator()

    result = cdg.create_prompts(["The elderly patient needs care."], attribute="age")
    assert "young_prompt" in result and "old_prompt" in result

    result = cdg.create_prompts(["The blind student uses a reader."], attribute="health-condition")
    assert "healthy_prompt" in result and "disabled_prompt" in result

    result = cdg.create_prompts(["The iranian student studied hard."], attribute="nationality")
    assert "american_prompt" in result and "german_prompt" in result

    result = cdg.create_prompts(["The overweight patient was checked."], attribute="physical-appearance")
    assert "fit_prompt" in result and "attractive_prompt" in result

    result = cdg.create_prompts(["The sikh prayed at noon."], attribute="religion")
    assert "christian_prompt" in result and "muslim_prompt" in result

    result = cdg.create_prompts(["The gay couple adopted a child."], attribute="sexual-orientation")
    assert "homosexual_prompt" in result and "heterosexual_prompt" in result

    result = cdg.create_prompts(["The wealthy donor contributed."], attribute="socioeconomic-class")
    assert "upper-class_prompt" in result and "working-class_prompt" in result
