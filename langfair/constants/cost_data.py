# Copyright 2024 CVS Health and/or one of its affiliates
#
# Copyright OpenAI 2015-2024
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
#
# The original work of OpenAI has been modified by CVS Health
# to include costs for only a subset of OpenAI models.

"""Storage place for constants related to cost information such as LLM invokation token cost."""

################################################################################
# A dictionary to store the cost information for different OpenAI models. It maps the model names to their respective input and output token costs.
################################################################################
COST_MAPPING = {
    "gpt-3.5-turbo-0613": {"input": 0.0000015, "output": 0.000002},
    "gpt-3.5-turbo-0301": {"input": 0.000002, "output": 0.000002},
    "gpt-3.5-turbo-16k-0613": {"input": 0.000003, "output": 0.000004},
    "gpt-3.5-turbo-16k-1106": {"input": 0.000001, "output": 0.000002},
    "gpt-3.5-turbo-0125": {"input": 0.0000005, "output": 0.0000015},
    "gpt-4-0314": {"input": 0.00003, "output": 0.00006},
    "gpt-4-0613": {"input": 0.00003, "output": 0.00006},
    "gpt-4-32k-0314": {"input": 0.00006, "output": 0.00012},
    "gpt-4-32k-0613": {"input": 0.00006, "output": 0.00012},
    "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
    "gpt-4-turbo-2024-04-09": {"input": 0.00001, "output": 0.00003},
}
FAILURE_MESSAGE = "Unable to get response"
TOKEN_COST_DATE = "10/21/2024"
