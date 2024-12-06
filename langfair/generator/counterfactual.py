# Copyright 2024 CVS Health and/or one of its affiliates
#
# Copyright 2023 OpenAI
#
# Licensed under the MIT License.
#
# The original work of OpenAI has been modified
# by CVS Health to include functionality for computing
# prompt and response token counts for OpenAI models.

import asyncio
import itertools
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import sacremoses
from nltk.tokenize import word_tokenize

from langfair.constants.word_lists import (
    FEMALE_WORDS,
    GENDER_NEUTRAL_WORDS,
    GENDER_TO_WORD_LISTS,
    MALE_WORDS,
    PERSON_WORDS,
    RACE_WORDS_NOT_REQUIRING_CONTEXT,
    RACE_WORDS_REQUIRING_CONTEXT,
)
from langfair.generator.generator import ResponseGenerator

# Constants for CounterfactualDatasetGenerator class
ALL_GENDER_WORDS = MALE_WORDS + FEMALE_WORDS
GENDER_MAPPING = {}
GENDER_NEUTRAL_MAPPING = {}
for i in range(0, len(MALE_WORDS)):
    GENDER_MAPPING[MALE_WORDS[i]] = FEMALE_WORDS[i]
    GENDER_MAPPING[FEMALE_WORDS[i]] = MALE_WORDS[i]
    GENDER_NEUTRAL_MAPPING[MALE_WORDS[i]] = GENDER_NEUTRAL_WORDS[i]
    GENDER_NEUTRAL_MAPPING[FEMALE_WORDS[i]] = GENDER_NEUTRAL_WORDS[i]

STRICT_RACE_WORDS = []
for rw in (
    RACE_WORDS_REQUIRING_CONTEXT
):  # Include token-pairs that indicate reference to the race of a person
    for pw in PERSON_WORDS:
        STRICT_RACE_WORDS.append(rw + " " + pw)

STRICT_RACE_WORDS.extend(
    RACE_WORDS_NOT_REQUIRING_CONTEXT
)  # Extend to include words that indicate race whether or not a person word follows
ALL_RACE_WORDS = RACE_WORDS_REQUIRING_CONTEXT + RACE_WORDS_NOT_REQUIRING_CONTEXT
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CounterfactualGenerator(ResponseGenerator):
    def __init__(
        self,
        langchain_llm: Any = None,
        suppressed_exceptions: Optional[
            Union[Tuple[BaseException], BaseException]
        ] = None,
        max_calls_per_min: Optional[int] = None,
    ) -> None:
        """
        Class for parsing and replacing protected attribute words.

        For the full list of gender and race words, refer to https://github.com/pages/cvs-health/langfair

        Parameters
        ----------
        langchain_llm : langchain llm object, default=None
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `langchain_llm` object.

        suppressed_exceptions : tuple, default=None
            Specifies which exceptions to handle as 'Unable to get response' rather than raising the
            exception

        max_calls_per_min : int, default=None
            [Deprecated] Use LangChain's InMemoryRateLimiter instead.
        """
        super().__init__(
            langchain_llm=langchain_llm,
            suppressed_exceptions=suppressed_exceptions,
            max_calls_per_min=max_calls_per_min,
        )
        self.attribute_to_word_lists = {
            "race": ALL_RACE_WORDS,
            "gender": ALL_GENDER_WORDS,
        }
        self.attribute_to_ref_dicts = {"gender": GENDER_TO_WORD_LISTS}
        self.gender_to_word_lists = GENDER_TO_WORD_LISTS
        self.cf_gender_mapping = GENDER_MAPPING
        self.gender_neutral_mapping = GENDER_NEUTRAL_MAPPING
        self.all_race_words = ALL_RACE_WORDS
        self.strict_race_words = STRICT_RACE_WORDS
        self.detokenizer = sacremoses.MosesDetokenizer("en")
        self.group_mapping = {
            "gender": ["male", "female"],
            "race": ["white", "black", "hispanic", "asian"],
        }

        try:
            word_tokenize("Check if this function can access the required corpus")
        except LookupError:
            nltk.download("punkt_tab")

    async def estimate_token_cost(
        self,
        tiktoken_model_name: str,
        prompts: List[str],
        attribute: str,
        example_responses: Optional[List[str]] = None,
        response_sample_size: int = 30,
        system_prompt: str = "You are a helpful assistant",
        count: int = 25,
    ) -> Dict[str, float]:
        """
        Estimates the token cost for a given list of prompts and (optionally) example responses.
        Note: This method is only compatible with GPT models.

        Parameters
        ----------
        prompts : list of strings
           A list of prompts

        tiktoken_model_name: str
           The name of the OpenAI model to use for token counting.

        attribute: str, either 'gender' or 'race'
            Specifies attribute to be used for counterfactual generation

        example_responses : list of strings, default=None
           A list of example responses. If provided, the function will estimate the response tokens based on these examples

        response_sample_size : int, default = 30.
           The number of responses to generate for cost estimation if `example_responses` is not provided.

        system_prompt : str, default="You are a helpful assistant."
           The system prompt to use.

        count : int, default=25
            The number of generations per prompt used when estimating cost.

        Returns
        -------
        dict
           A dictionary containing the estimated token costs, including prompt token cost, completion token cost,
           and total token cost.
        """
        prompts = list(prompts)
        parse_result = self.parse_texts(texts=prompts, attribute=attribute)
        prompts_sub = [prompts[i] for i in range(len(parse_result)) if parse_result[i]]
        result = await ResponseGenerator().estimate_token_cost(
            tiktoken_model_name=tiktoken_model_name,
            prompts=prompts_sub,
            example_responses=example_responses,
            response_sample_size=response_sample_size,
            system_prompt=system_prompt,
            count=count,
        )
        return {
            key: value * len(self.group_mapping[attribute])
            for key, value in result.items()
        }

    def parse_texts(
        self,
        texts: List[str],
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Parses a list of texts for protected attribute words

        Parameters
        ----------
        texts : list of strings
            A list of texts to be parsed for protected attribute words

        attribute : {'race','gender'}, default=None
            Specifies what to parse for among race words and gender words. Must be specified
            if custom_list is None

        custom_list : List[str], default=None
            Custom list of tokens to use for parsing prompts. Must be provided if attribute is None.

        Returns
        -------
        list
            List of length `len(texts)` with each element being a list of identified protected
            attribute words in provided text
        """
        assert not (custom_list and attribute), """
        Either custom_list or attribute must be None.
        """
        assert custom_list or attribute in ["race", "gender"], """
        If custom_list is None, attribute must be 'race' or 'gender'.
        """
        result = []
        for text in texts:
            result.append(
                self._token_parser(
                    text=text, attribute=attribute, custom_list=custom_list
                )
            )
        return result

    def create_prompts(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        custom_dict: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Creates prompts by counterfactual substitution

        Parameters
        ----------
        prompts : List[str]
            A list of prompts on which counterfactual substitution and response generation will be done

        attribute : {'gender', 'race'}, default=None
            Specifies whether to use race or gender for counterfactual substitution. Must be provided if
            custom_dict is None.

        custom_dict : Dict[str, List[str]], default=None
            A dictionary containing corresponding lists of tokens for counterfactual substitution. Keys
            should correspond to groups. Must be provided if attribute is None. For example:
            {'male': ['he', 'him', 'woman'], 'female': ['she', 'her', 'man']}

        Returns
        -------
        dict
            Dictionary containing counterfactual prompts
        """
        assert not (custom_dict and attribute), """
        Either custom_dict or attribute must be None.
        """
        assert custom_dict or attribute in [
            "gender",
            "race",
        ], "If custom_dict is None, attribute must be 'gender' or 'race'."

        custom_list = (
            list(itertools.chain(*custom_dict.values())) if custom_dict else None
        )

        prompts, attribute_words = self._subset_prompts(
            prompts=prompts, attribute=attribute, custom_list=custom_list
        )

        if attribute == "race":
            prompts_dict = {
                race + "_prompt": self._counterfactual_sub_race(
                    texts=prompts, target_race=race
                )
                for race in self.group_mapping[attribute]
            }

        else:
            if custom_dict:
                ref_dict = custom_dict
            elif attribute == "gender":
                ref_dict = self.attribute_to_ref_dicts[attribute]

            prompts_dict = {key + "_prompt": [] for key in ref_dict}
            for prompt in prompts:
                counterfactual_prompts = self._sub_from_dict(
                    ref_dict=ref_dict, text=prompt
                )
                self.counterfactual_prompts = counterfactual_prompts
                for key in counterfactual_prompts:
                    prompts_dict[key + "_prompt"].append(counterfactual_prompts[key])

        prompts_dict["original_prompt"] = prompts
        prompts_dict["attribute_words"] = [
            attr_word for attr_word in attribute_words if len(attr_word) > 0
        ]
        return prompts_dict

    def neutralize_tokens(
        self, texts: List[str], attribute: str = "gender"
    ) -> List[str]:
        """
        Neutralize gender and race words contained in a list of texts. Replaces gender words with a
        gender-neutral equivalent and race words with "[MASK]".

        Parameters
        ----------
        texts : List[str]
            A list of texts on which gender or race neutralization will occur

        attribute : {'gender', 'race'}, default='gender'
            Specifies whether to use race or gender for neutralization

        Returns
        -------
        list
            List of texts neutralized for race or gender
        """
        assert attribute in [
            "gender",
            "race",
        ], "Only gender and race attributes are supported."
        if attribute == "gender":
            return [self._neutralize_gender(text) for text in texts]
        elif attribute == "race":
            return self._counterfactual_sub_race(texts=texts, target_race="[MASK]")

    async def generate_responses(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        count: int = 25,
        custom_dict: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Creates prompts by counterfactual substitution and generates responses asynchronously

        Parameters
        ----------
        prompts : list of strings
            A list of prompts on which counterfactual substitution and response generation will be done

        attribute : {'gender', 'race'}, default=None
            Specifies whether to use race or gender for counterfactual substitution. Must be provided if
            custom_dict is None.

        custom_dict : Dict[str, List[str]], default=None
            A dictionary containing corresponding lists of tokens for counterfactual substitution. Keys
            should correspond to groups. Must be provided if attribute is None. For example:
            {'male': ['he', 'him', 'woman'], 'female': ['she', 'her', 'man']}

        system_prompt : str, default="You are a helpful assistant."
            Specifies system prompt for generation

        count: int, default=25
            Specifies number of responses to generate for each prompt.

        Returns
        ----------
        dict
            A dictionary with two keys: 'data' and 'metadata'.
            'data' : dict
                A dictionary containing the prompts and responses.
            'metadata' : dict
                A dictionary containing metadata about the generation process.
                'non_completion_rate' : float
                    The rate at which the generation process did not complete.
                'temperature' : float
                    The temperature parameter used in the generation process.
                'count' : int
                    The count of prompts used in the generation process.
                'system_prompt' : str
                    The system prompt used for generating responses
        """
        if self.llm.temperature == 0:
            assert count == 1, "temperature must be greater than 0 if count > 1"
        self.count = count

        # create counterfactual prompts
        groups = self.group_mapping[attribute] if attribute else custom_dict.keys()
        prompts_dict = self.create_prompts(
            prompts=prompts,
            attribute=attribute,
            custom_dict=custom_dict,
        )

        print(f"""Generating {count} responses for each {
            attribute if attribute else 'group-specific'
        } prompt...""")

        # generate responses with async
        responses_dict, duplicated_prompts_dict = {}, {}
        chain = self._setup_langchain(system_prompt=system_prompt)
        for group in groups:
            prompt_key = group + "_prompt"
            # start = time.time()
            # generate with async
            (
                tasks,
                duplicated_prompts_dict[prompt_key],
            ) = self._create_tasks(chain=chain, prompts=prompts_dict[prompt_key])
            responses_dict[group + "_response"] = await asyncio.gather(*tasks)
            # stop = time.time()

        non_completion_rate = len(
            [
                i
                for i, vals in enumerate(zip(responses_dict.values()))
                if self.failure_message in vals
            ]
        ) / len(list(responses_dict.values())[0])

        print("Responses successfully generated!")
        return {
            "data": {
                **duplicated_prompts_dict,
                **responses_dict,
            },
            "metadata": {
                "non_completion_rate": non_completion_rate,
                "system_prompt": system_prompt,
                "temperature": self.llm.temperature,
                "count": self.count,
                "groups": groups,
                "original_prompts": prompts_dict["original_prompt"],
                "attribute_words": prompts_dict["attribute_words"],
            },
        }

    def _subset_prompts(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[List[str]]]:
        """Subset prompts that contain protected attribute words"""
        attribute_to_print = (
            "Protected attribute" if not attribute else attribute.capitalize()
        )
        attribute_words = self.parse_texts(
            texts=prompts, attribute=attribute, custom_list=custom_list
        )
        prompts_subset = [
            prompt for i, prompt in enumerate(prompts) if attribute_words[i]
        ]
        assert len(prompts_subset) > 0, f"""
        Provided prompts do not contain any {attribute_to_print} words.
        """
        print(f"{attribute_to_print} words found in {len(prompts_subset)} prompts.")
        return prompts_subset, attribute_words

    def _counterfactual_sub_race(
        self,
        texts: List[str],
        target_race: str,
    ) -> List[str]:
        """Implements counterfactual substitution"""
        new_texts = []
        for text in texts:
            # race replacement
            new_text = self._replace_race(text, target_race)
            new_texts.append(new_text)
        return new_texts

    def _neutralize_gender(self, text: str) -> str:
        """Replaces gender words with target gender words"""
        raw_tokens = word_tokenize(text)
        lower_tokens = word_tokenize(text.lower())
        neutral_tokens = [
            self.gender_neutral_mapping[lower]
            if (lower in self.attribute_to_word_lists["gender"])
            else token
            for token, lower in zip(raw_tokens, lower_tokens)
        ]
        return self.detokenizer.detokenize(neutral_tokens)

    def _token_parser(
        self,
        text: str,
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
    ) -> List[str]:
        """Helper function for parsing tokens"""
        tokens = word_tokenize(str(text).lower())
        if attribute == "race":
            return self._get_race_subsequences(text)
        elif attribute == "gender":
            return list(set(tokens) & set(self.attribute_to_word_lists[attribute]))
        elif custom_list:
            return list(set(tokens) & set(custom_list))

    def _sub_from_dict(
        self, ref_dict: Dict[str, List[str]], text: str
    ) -> Dict[str, List[str]]:
        """
        Creates counterfactual variations based on a dictionary of reference lists.
        """
        ref_dict = {key: [t.lower() for t in val] for key, val in ref_dict.items()}
        lower_tokens = word_tokenize(text)

        ref_values = {
            val: idx for key in ref_dict for idx, val in enumerate(ref_dict[key])
        }
        output_dict = {key: [None] * len(lower_tokens) for key in ref_dict}
        for key in ref_dict.keys():
            for i, element in enumerate(lower_tokens):
                output_dict[key][i] = (
                    ref_dict[key][ref_values[element]]
                    if element in ref_values
                    else element
                )
            output_dict[key] = self.detokenizer.detokenize(output_dict[key])

        return output_dict

    ################################################################################
    # Class for protected attribute scanning and replacing protected attribute words
    ################################################################################
    @staticmethod
    def _get_race_subsequences(text: str) -> List[str]:
        """Used to check for string sequences"""
        seq = text.lower()
        return [subseq for subseq in STRICT_RACE_WORDS if subseq in seq]

    @staticmethod
    def _replace_race(text: str, target_race: str) -> str:
        """Replaces text with a target word"""
        seq = text.lower()
        race_replacement_mapping = {}
        for rw in RACE_WORDS_REQUIRING_CONTEXT:
            for pw in PERSON_WORDS:
                key = rw + " " + pw
                race_replacement_mapping[key] = target_race + " " + pw
        for rw in RACE_WORDS_NOT_REQUIRING_CONTEXT:
            race_replacement_mapping[rw] = target_race

        for subseq in STRICT_RACE_WORDS:
            seq = seq.replace(subseq, race_replacement_mapping[subseq])
        return seq
