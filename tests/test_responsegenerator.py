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

import itertools

import pytest
from langchain_openai import AzureChatOpenAI

from langfair.generator import ResponseGenerator


@pytest.mark.asyncio
async def test_generator(monkeypatch):
    count = 3
    MOCKED_PROMPTS = ["Prompt 1", "Prompt 2", "Prompt 3"]
    MOCKED_DUPLICATE_PROMPTS = [
        prompt for prompt, i in itertools.product(MOCKED_PROMPTS, range(count))
    ]
    MOCKED_RESPONSES = [
        "Mocked response 1",
        "Mocked response 2",
        "Unable to get response",
    ]
    MOCKED_DUPLICATED_RESPONSES = [
        prompt for prompt, i in itertools.product(MOCKED_RESPONSES, range(count))
    ]
    MOCKED_RESPONSE_DICT = dict(zip(MOCKED_PROMPTS, MOCKED_RESPONSES))

    async def mock_async_api_call(prompt, count, *args, **kwargs):
        return [MOCKED_RESPONSE_DICT[prompt]] * count

    mock_object = AzureChatOpenAI(
        deployment_name="YOUR-DEPLOYMENT",
        temperature=1,
        api_key="SECRET_API_KEY",
        api_version="2024-05-01-preview",
        azure_endpoint="https://mocked.endpoint.com",
    )

    generator_object = ResponseGenerator(langchain_llm=mock_object)

    monkeypatch.setattr(generator_object, "_async_api_call", mock_async_api_call)
    data = await generator_object.generate_responses(
        prompts=MOCKED_PROMPTS, count=count
    )

    cost = await generator_object.estimate_token_cost(
        tiktoken_model_name="gpt-3.5-turbo-16k-0613",
        prompts=MOCKED_DUPLICATE_PROMPTS,
        example_responses=MOCKED_RESPONSES[:3],
        count=count,
    )

    assert data["data"]["response"] == MOCKED_DUPLICATED_RESPONSES
    assert data["metadata"]["non_completion_rate"] == 1 / 3
    assert cost == {
        "Estimated Prompt Token Cost (USD)": 0.001539,
        "Estimated Completion Token Cost (USD)": 0.000504,
        "Estimated Total Token Cost (USD)": 0.002043,
    }
