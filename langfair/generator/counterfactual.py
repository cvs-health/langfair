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
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import sacremoses

# Import for LLM-based FTU checking
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.system import SystemMessage
from nltk.tokenize import word_tokenize

from langfair.constants.cost_data import FAILURE_MESSAGE
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
STRICT_RACE_WORDS = list(set(STRICT_RACE_WORDS))
ALL_RACE_WORDS = RACE_WORDS_REQUIRING_CONTEXT + RACE_WORDS_NOT_REQUIRING_CONTEXT
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CounterfactualGenerator(ResponseGenerator):
    def __init__(
        self,
        langchain_llm: Any = None,
        suppressed_exceptions: Optional[
            Union[Tuple[BaseException], BaseException, Dict[BaseException, str]]
        ] = None,
        use_n_param: bool = False,
        max_calls_per_min: Optional[int] = None,
    ) -> None:
        """
        Class for parsing and replacing protected attribute words.

        For the full list of gender and race words, refer to https://github.com/pages/cvs-health/langfair

        Parameters
        ----------
        langchain_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `langchain_llm` object.

        suppressed_exceptions : tuple or dict, default=None
            If a tuple, specifies which exceptions to handle as 'Unable to get response' rather than raising the
            exception. If a dict, enables users to specify exception-specific failure messages with keys being subclasses
            of BaseException

        use_n_param : bool, default=False
            Specifies whether to use `n` parameter for `BaseChatModel`. Not compatible with all
            `BaseChatModel` classes. If used, it speeds up the generation process substantially when count > 1.

        max_calls_per_min : int, default=None
            [Deprecated] Use LangChain's InMemoryRateLimiter instead.
        """
        super().__init__(
            langchain_llm=langchain_llm,
            suppressed_exceptions=suppressed_exceptions,
            max_calls_per_min=max_calls_per_min,
        )
        self.use_n_param = use_n_param
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
        self._validate_attributes(attribute=attribute, custom_list=custom_list)
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
        self._validate_attributes(
            attribute=attribute, custom_dict=custom_dict, for_parsing=False
        )

        # Use traditional static method
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

    def create_prompts_from_llm_terms(
        self,
        prompts: List[str],
        llm_detected_terms: List[List[str]],
        attribute: Optional[str] = None,
        custom_dict: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Creates prompts by counterfactual substitution using LLM-detected terms.

        This method works with terms detected by the LLM-based FTU checker,
        allowing for more flexible counterfactual generation.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts for counterfactual substitution

        llm_detected_terms : List[List[str]]
            List of detected terms for each prompt (from check_ftu_llm)

        attribute : {'gender', 'race'}, default=None
            Specifies attribute type for substitution logic

        custom_dict : Dict[str, List[str]], default=None
            Custom substitution dictionary

        Returns
        -------
        dict
            Dictionary containing counterfactual prompts
        """
        # Filter prompts that have detected terms
        filtered_prompts = []
        filtered_terms = []

        for prompt, terms in zip(prompts, llm_detected_terms):
            if terms:  # Only include prompts with detected terms
                filtered_prompts.append(prompt)
                filtered_terms.append(terms)

        if not filtered_prompts:
            # No prompts with detected terms - return empty structure matching expected groups
            groups = (
                self.group_mapping[attribute] if attribute else list(custom_dict.keys())
            )
            result = {
                "original_prompt": [],
                "attribute_words": [],
            }
            # Add empty lists for each group
            for group in groups:
                result[f"{group}_prompt"] = []
            return result

        if attribute == "race":
            # For race, create counterfactuals for each race group
            prompts_dict = {
                race + "_prompt": self._counterfactual_sub_race_with_terms(
                    texts=filtered_prompts,
                    detected_terms=filtered_terms,
                    target_race=race,
                )
                for race in self.group_mapping[attribute]
            }
        else:
            # For gender or custom attributes
            if custom_dict:
                ref_dict = custom_dict
            elif attribute == "gender":
                ref_dict = self.attribute_to_ref_dicts[attribute]
            else:
                # Fallback: use detected terms to create a simple substitution
                ref_dict = {"group1": [], "group2": []}

            prompts_dict = {key + "_prompt": [] for key in ref_dict}
            for prompt, terms in zip(filtered_prompts, filtered_terms):
                counterfactual_prompts = self._sub_from_dict_with_terms(
                    ref_dict=ref_dict, text=prompt, detected_terms=terms
                )
                for key in counterfactual_prompts:
                    prompts_dict[key + "_prompt"].append(counterfactual_prompts[key])

        prompts_dict["original_prompt"] = filtered_prompts
        prompts_dict["attribute_words"] = filtered_terms
        return prompts_dict

    def _counterfactual_sub_race_with_terms(
        self,
        texts: List[str],
        detected_terms: List[List[str]],
        target_race: str,
    ) -> List[str]:
        """Implements counterfactual substitution using LLM-detected race terms."""
        new_texts = []
        for text, terms in zip(texts, detected_terms):
            new_text = text
            # Replace each detected race term
            for term in terms:
                if term in RACE_WORDS_NOT_REQUIRING_CONTEXT:
                    # Direct replacement for standalone race words that don't need context
                    new_text = re.sub(
                        rf"\b{re.escape(term)}\b",
                        target_race,
                        new_text,
                        flags=re.IGNORECASE,
                    )
                elif any(
                    term.startswith(rw + " ") for rw in RACE_WORDS_REQUIRING_CONTEXT
                ):
                    # Handle race + person combinations (e.g., "white male" -> "hispanic male")
                    parts = term.split(" ", 1)
                    if len(parts) == 2:
                        race_part, person_part = parts
                        replacement = f"{target_race} {person_part}"
                        new_text = re.sub(
                            rf"\b{re.escape(term)}\b",
                            replacement,
                            new_text,
                            flags=re.IGNORECASE,
                        )
                elif term in RACE_WORDS_REQUIRING_CONTEXT:
                    # Handle standalone context words (e.g., "asian" -> "hispanic")
                    # Even though they "require context", if the LLM detected them standalone,
                    # we should still replace them
                    new_text = re.sub(
                        rf"\b{re.escape(term)}\b",
                        target_race,
                        new_text,
                        flags=re.IGNORECASE,
                    )
            new_texts.append(new_text)
        return new_texts

    def _sub_from_dict_with_terms(
        self, ref_dict: Dict[str, List[str]], text: str, detected_terms: List[str]
    ) -> Dict[str, str]:
        """Creates counterfactual variations using detected explicit gender/race terms only.

        This method focuses on explicit terms (he/she, male/female, caucasian/black, etc.)
        like the static method, but uses LLM-detected terms for better robustness.
        """
        # Filter out names - we only want explicit gender/race terms like the static method
        explicit_terms = [term for term in detected_terms if ":name:" not in term]

        if not explicit_terms:
            # No explicit terms found, return original text for all groups
            return {group: text for group in ref_dict.keys()}

        # Use traditional word replacement with the detected explicit terms
        return self._sub_from_dict_with_detected_terms(
            ref_dict=ref_dict, text=text, detected_terms=explicit_terms
        )

    def _sub_from_dict_with_detected_terms(
        self, ref_dict: Dict[str, List[str]], text: str, detected_terms: List[str]
    ) -> Dict[str, str]:
        """Substitute detected explicit terms using the reference dictionary.

        This mimics the static method behavior but uses LLM-detected terms.
        """
        variations = {}

        for group_key, word_list in ref_dict.items():
            new_text = text

            # For each detected term, try to replace it with words from the current group
            for term in detected_terms:
                term_lower = term.lower()

                # Find the best replacement from the word list
                # This mimics how the static method maps words
                replacement = None

                # For gender, try to find direct mappings
                if group_key in ["male", "female"]:
                    if term_lower in GENDER_MAPPING:
                        # Use the opposite gender mapping like static method
                        if group_key == "male" and term_lower in FEMALE_WORDS:
                            replacement = GENDER_MAPPING[term_lower]  # female -> male
                        elif group_key == "female" and term_lower in MALE_WORDS:
                            replacement = GENDER_MAPPING[term_lower]  # male -> female
                        elif group_key == "male" and term_lower in MALE_WORDS:
                            replacement = term_lower  # keep male terms in male group
                        elif group_key == "female" and term_lower in FEMALE_WORDS:
                            replacement = (
                                term_lower  # keep female terms in female group
                            )

                # If we found a replacement, apply it
                if replacement:
                    # Use word boundaries for precise replacement
                    new_text = re.sub(
                        rf"\b{re.escape(term)}\b",
                        replacement,
                        new_text,
                        flags=re.IGNORECASE,
                    )

            variations[group_key] = new_text

        return variations

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
        llm_ftu: Optional["BaseChatModel"] = None,
        llm_ftu_threshold: float = 0.5,
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

        llm_ftu : Optional[BaseChatModel], default=None
            LLM-based FTU checking:
            - None: No automatic FTU checking, uses static approach (default)
            - BaseChatModel: Uses the provided LangChain model for automatic FTU checking
            If provided, will automatically run LLM-based FTU check before generating counterfactuals.

        llm_ftu_threshold : float, default=0.3
            Confidence threshold for LLM-based FTU checking. Only used when llm_ftu is not None.

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

                'prompt' : list
                    A list of prompts.
                'response' : list
                    A list of responses corresponding to the prompts.

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
        self._update_count(count)
        self.system_message = SystemMessage(system_prompt)

        # create counterfactual prompts
        groups = self.group_mapping[attribute] if attribute else custom_dict.keys()

        if llm_ftu is not None:
            # Validate LLM type early in the flow
            if not isinstance(llm_ftu, BaseChatModel):
                raise ValueError(
                    f"llm_ftu must be a LangChain BaseChatModel, got {type(llm_ftu).__name__}. "
                    "Example: from langchain_google_vertexai import ChatVertexAI; llm_ftu = ChatVertexAI()"
                )

            print("Running LLM-based FTU check...")

            # Run LLM-based FTU check and get detected terms directly
            detected_prompts, detected_terms = self._run_llm_ftu_check(
                prompts=prompts,
                attribute=attribute,
                llm=llm_ftu,
                llm_threshold=llm_ftu_threshold,
            )

            # Use LLM-detected terms for counterfactual generation
            prompts_dict = self.create_prompts_from_llm_terms(
                prompts=detected_prompts,
                llm_detected_terms=detected_terms,
                attribute=attribute,
                custom_dict=custom_dict,
            )
        else:
            # Use traditional static method
            prompts_dict = self.create_prompts(
                prompts=prompts,
                attribute=attribute,
                custom_dict=custom_dict,
            )

        # Check if there are any prompts to generate responses for
        has_prompts_to_generate = any(
            len(prompts_dict.get(f"{group}_prompt", [])) > 0 for group in groups
        )

        if has_prompts_to_generate:
            print(
                f"""Generating {count} responses for each {
                    attribute if attribute else "group-specific"
                } prompt..."""
            )

            # generate responses with async
            responses_dict, duplicated_prompts_dict = {}, {}
            for group in groups:
                prompt_key = group + "_prompt"
                # start = time.time()
                # generate with async
                (
                    tasks,
                    duplicated_prompts_dict[prompt_key],
                ) = self._create_tasks(prompts=prompts_dict[prompt_key])
                tmp_response_list = await asyncio.gather(*tasks)

                tmp_responses = []
                for response in tmp_response_list:
                    tmp_responses.extend(response)
                responses_dict[group + "_response"] = self._enforce_strings(
                    tmp_responses
                )
                # stop = time.time()

            print("Responses successfully generated!")
        else:
            print(
                "No counterfactual prompts to generate responses for - FTU satisfied with no explicit terms detected."
            )
            # Initialize empty response structures
            responses_dict, duplicated_prompts_dict = {}, {}
            for group in groups:
                responses_dict[group + "_response"] = []
                duplicated_prompts_dict[group + "_prompt"] = []
        # Determine FTU method used for metadata
        ftu_method_used = "llm" if llm_ftu is not None else "static"

        return {
            "data": {
                **duplicated_prompts_dict,
                **responses_dict,
            },
            "metadata": {
                "non_completion_rate": self._calc_noncompletion_rate(responses_dict),
                "system_prompt": system_prompt,
                "temperature": self.llm.temperature,
                "count": self.count,
                "groups": groups,
                "original_prompts": prompts_dict["original_prompt"],
                "attribute_words": prompts_dict["attribute_words"],
                "ftu_method": ftu_method_used,
                "auto_ftu": llm_ftu is not None,
            },
        }

    def check_ftu(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
        subset_prompts: bool = True,
        llm: Optional["BaseChatModel"] = None,
        llm_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Checks for fairness through unawarenss (FTU) based on a list of prompts and a specified protected
        attribute. Supports both static word list and LLM-based detection methods.

        Parameters
        ----------
        prompts : list of strings
            A list of prompts to be parsed for protected attribute words

        attribute : {'race','gender'}, default=None
            Specifies what to parse for among race words and gender words. Must be specified
            if custom_list is None

        custom_list : List[str], default=None
            Custom list of tokens to use for parsing prompts. Must be provided if attribute is None.

        subset_prompts : bool, default=True
            Indicates whether to return all prompts or only those containing attribute words

        llm : Optional[BaseChatModel], default=None
            Controls the detection method:
            - None: Uses static word lists (default, fastest)
            - BaseChatModel: Uses the provided LangChain model (e.g., ChatVertexAI, ChatOpenAI)

        llm_threshold : float, default=0.3
            Confidence threshold for LLM-based classification (0.0 to 1.0). Default of 0.3 is
            intentionally low to err on the side of caution in bias detection - false negatives
            are more problematic than false positives in fairness contexts. Only used when llm is not None.

        Returns
        -------
        dict
            A dictionary with two keys: 'data' and 'metadata'.

            'data' : dict
                A dictionary containing the prompts and the attribute words they contain.

                'prompt' : list
                    A list of prompts.

                'attribute_words' : list
                    A list of attribute_words in each prompt.

            'metadata' : dict
                A dictionary containing metadata related to FTU.

                'ftu_satisfied' : boolean
                    Boolean indicator of whether or not prompts satisfy FTU

                'filtered_prompt_count' : int
                    The number of prompts that satisfy FTU.

                'method' : str
                    Detection method used ('static' or 'llm')
        """
        # Route to appropriate detection method
        if llm is None:
            return self._check_ftu_static(
                prompts=prompts,
                attribute=attribute,
                custom_list=custom_list,
                subset_prompts=subset_prompts,
            )
        else:
            # Validate LLM type early
            if not isinstance(llm, BaseChatModel):
                raise ValueError(
                    f"llm parameter must be either None (for static method) or a LangChain BaseChatModel instance, "
                    f"got {type(llm).__name__}. "
                    "Example: from langchain_google_vertexai import ChatVertexAI; llm = ChatVertexAI()"
                )

            return self._check_ftu_llm(
                prompts=prompts,
                attribute=attribute,
                custom_list=custom_list,
                subset_prompts=subset_prompts,
                llm=llm,
                llm_threshold=llm_threshold,
            )

    def _check_ftu_static(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
        subset_prompts: bool = True,
    ) -> Dict[str, Any]:
        """Internal method for static word list FTU checking."""
        self._validate_attributes(attribute=attribute, custom_list=custom_list)
        attribute_to_print = (
            "Protected attribute" if not attribute else attribute.capitalize()
        )
        attribute_words = self.parse_texts(
            texts=prompts,
            attribute=attribute,
            custom_list=custom_list,
        )
        prompts_subset = [
            prompt for i, prompt in enumerate(prompts) if attribute_words[i]
        ]
        attribute_words_subset = [
            aw for i, aw in enumerate(attribute_words) if attribute_words[i]
        ]

        n_prompts_with_attribute_words = len(prompts_subset)
        ftu_satisfied = n_prompts_with_attribute_words > 0
        ftu_text = " not " if ftu_satisfied else " "

        ftu_print = f"FTU is{ftu_text}satisfied."
        print(
            f"{attribute_to_print} words found in {len(prompts_subset)} prompts. {ftu_print}"
        )

        return {
            "data": {
                "prompt": prompts_subset if subset_prompts else prompts,
                "attribute_words": attribute_words_subset
                if subset_prompts
                else attribute_words,
            },
            "metadata": {
                "ftu_satisfied": ftu_satisfied,
                "n_prompts_with_attribute_words": n_prompts_with_attribute_words,
                "attribute": attribute,
                "custom_list": custom_list,
                "subset_prompts": subset_prompts,
                "method": "static",
            },
        }

    def _check_ftu_llm(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
        subset_prompts: bool = True,
        llm: "BaseChatModel" = None,
        llm_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Internal method for LLM-based FTU checking using LangChain BaseChatModel."""
        # Note: Type validation is handled earlier in generate_responses()

        # For LLM method, we need to handle both single attribute and custom_list cases
        if custom_list:
            # For custom lists, we'll treat it as a generic attribute
            attributes = ["custom"]
        elif attribute:
            attributes = [attribute]
        else:
            # Default to both race and gender
            attributes = ["race", "gender"]

        detected_prompts = []
        detected_attributes = []

        for prompt in prompts:
            if custom_list:
                # For custom lists, use a simplified approach
                prompt_attributes = self._check_custom_attributes_llm(
                    prompt,
                    custom_list,
                    llm_threshold,
                    llm,
                )
            else:
                # Use LangChain model for detection
                prompt_attributes = self._check_attributes_with_langchain(
                    prompt, attributes, llm_threshold, llm
                )

            if prompt_attributes:
                detected_prompts.append(prompt)
                detected_attributes.append(prompt_attributes)
            elif not subset_prompts:
                detected_prompts.append(prompt)
                detected_attributes.append([])

        n_prompts_with_attributes = len(
            [attrs for attrs in detected_attributes if attrs]
        )
        ftu_satisfied = n_prompts_with_attributes > 0

        attribute_to_print = (
            "Protected attribute" if not attribute else attribute.capitalize()
        )
        ftu_text = " not " if ftu_satisfied else " "
        print(
            f"LLM-based FTU check: {n_prompts_with_attributes} prompts contain {attribute_to_print.lower()} words"
        )
        print(f"FTU is{ftu_text}satisfied.")

        return {
            "data": {
                "prompt": detected_prompts,
                "attribute_words": detected_attributes,
            },
            "metadata": {
                "ftu_satisfied": ftu_satisfied,
                "n_prompts_with_attribute_words": n_prompts_with_attributes,
                "method": "llm",
                "llm_threshold": llm_threshold,
                "attribute": attribute,
                "custom_list": custom_list,
                "subset_prompts": subset_prompts,
            },
        }

    def _check_custom_attributes_llm(
        self,
        text: str,
        custom_list: List[str],
        threshold: float,
        langchain_llm: Optional[Any] = None,
    ) -> List[str]:
        """Check for custom attributes using LLM."""
        # For custom lists, do a simple check if any terms from the list are mentioned
        detected_terms = []

        if langchain_llm:
            # Use LangChain LLM to check for custom terms
            system_prompt = f"""You are an expert at detecting specific terms in text.

Your task is to identify if any of these terms appear in the given text: {", ".join(custom_list)}

Instructions:
1. Look for any mentions of the specified terms in the text
2. If you find any terms, list the EXACT words or phrases from the text
3. If no terms are found, respond with "NONE"
4. Be precise and consider context
5. Return only the specific terms found, separated by commas if multiple"""

            human_prompt = f"Text to analyze: {text}"

            from langchain_core.messages import HumanMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            try:
                response = langchain_llm.invoke(messages)
                response_text = response.content.strip()

                if response_text.upper() != "NONE":
                    terms = [term.strip().lower() for term in response_text.split(",")]
                    detected_terms.extend(terms)
            except Exception as e:
                warnings.warn(f"LLM custom term detection failed: {e}")
        else:
            # For HuggingFace, fall back to simple token matching
            tokens = word_tokenize(text.lower())
            for term in custom_list:
                if term.lower() in tokens:
                    detected_terms.append(term.lower())

        return detected_terms

    def _subset_prompts(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Helper function to subset prompts that contain protected attribute words and also
        return the full set of parsing results
        """
        attribute_to_print = (
            "Protected attribute" if not attribute else attribute.capitalize()
        )
        attribute_words = self.parse_texts(
            texts=prompts, attribute=attribute, custom_list=custom_list
        )
        prompts_subset = [
            prompt for i, prompt in enumerate(prompts) if attribute_words[i]
        ]

        if len(prompts_subset) == 0:
            # Handle the case where no attribute words are found
            print(
                f"No {attribute_to_print.lower()} words found in prompts - FTU satisfied."
            )
            return [], attribute_words

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
        lower_tokens = word_tokenize(text.lower())

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

    def _calc_noncompletion_rate(self, responses_dict: Dict[str, Any]) -> float:
        """Computes noncompletion rate"""
        if isinstance(self.suppressed_exceptions, Dict):
            non_completion_rate = len(
                [
                    i
                    for i, vals in enumerate(zip(responses_dict.values()))
                    if any(
                        value in vals for value in self.suppressed_exceptions.values()
                    )
                    or FAILURE_MESSAGE in vals
                ]
            ) / len(list(responses_dict.values())[0])
        else:
            non_completion_rate = len(
                [
                    i
                    for i, vals in enumerate(zip(responses_dict.values()))
                    if FAILURE_MESSAGE in vals
                ]
            ) / len(list(responses_dict.values())[0])
        return non_completion_rate

    @staticmethod
    def _get_race_subsequences(text: str) -> List[str]:
        """
        Used to check for race word sequences using word boundaries to avoid false positives.
        """
        text_lower = text.lower()
        found_words = []

        # Check words requiring context (e.g., "black person", "white male")
        for rw in RACE_WORDS_REQUIRING_CONTEXT:
            for pw in PERSON_WORDS:
                phrase = rw + " " + pw
                # Use word boundaries to match complete phrases
                pattern = rf"\b{re.escape(rw)}\s+{re.escape(pw)}\b"
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_words.append(phrase)

        # Check words not requiring context (e.g., "caucasian", "hispanic")
        for rw in RACE_WORDS_NOT_REQUIRING_CONTEXT:
            # Use word boundaries to match complete words
            pattern = rf"\b{re.escape(rw)}\b"
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_words.append(rw)

        return found_words

    def _check_attributes_with_langchain(
        self, text: str, attributes: List[str], threshold: float, langchain_llm: Any
    ) -> List[str]:
        """Check for protected attributes using LangChain LLM and extract specific terms."""
        detected_terms = []

        for attribute in attributes:
            if attribute == "gender":
                # Focus on explicit gender terms only (no names) - mimics static method
                gender_terms = self._detect_gender_explicit_terms(text, langchain_llm)
                detected_terms.extend(gender_terms)
            elif attribute == "race":
                # Focus on explicit race terms only - mimics static method
                race_terms = self._detect_race_explicit_terms(text, langchain_llm)
                detected_terms.extend(race_terms)
            else:
                raise ValueError(
                    f"Unsupported attribute '{attribute}'. Only 'gender' and 'race' are supported."
                )

        return detected_terms

    def _detect_gender_explicit_terms(self, text: str, langchain_llm: Any) -> List[str]:
        """Detect explicit gender terms only (no names) - mimics static method behavior."""
        detected_terms = []

        system_prompt = """You are an expert at detecting explicit gender references in text.

Your task is to identify explicit gender-related words (pronouns, titles, descriptors) in the given text.

Instructions:
1. Look ONLY for explicit gender words like: he, she, him, her, his, hers, man, woman, male, female, boy, girl, gentleman, lady, etc.
2. Do NOT include names (John, Jennifer, etc.) - only explicit gender terms that have clear interchangeable meanings between male and female
3. If you find gender words, list the EXACT words from the text
4. If no explicit gender words are found, respond with "NONE"
5. Return only the specific words, separated by commas if multiple
6. Focus on words that would be in a traditional gender word list

Examples:
- "She is a doctor" → "she"
- "The man walked to his car" → "man, his"
- "Jennifer is better than John" → "NONE" (names don't count)
- "The male patient" → "male"
- "A woman's perspective" → "woman's"
"""

        human_prompt = f"Text to analyze: {text}"

        from langchain_core.messages import HumanMessage

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
            response = langchain_llm.invoke(messages)
            response_text = response.content.strip()

            if response_text.upper() != "NONE":
                words = [term.strip().lower() for term in response_text.split(",")]
                detected_terms.extend(words)
        except Exception as e:
            warnings.warn(f"LLM gender term detection failed: {e}")

        return detected_terms

    def _run_llm_ftu_check(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        llm: "BaseChatModel" = None,
        llm_threshold: float = 0.3,
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Run LLM-based FTU check and return detected prompts and terms directly.

        Returns:
            Tuple of (prompts_with_terms, detected_terms_per_prompt)
        """
        # Detect terms for each prompt
        detected_terms_per_prompt = []
        prompts_with_terms = []

        for prompt in prompts:
            # Use LangChain model
            if attribute == "gender":
                terms = self._detect_gender_explicit_terms(prompt, llm)
            elif attribute == "race":
                terms = self._detect_race_explicit_terms(prompt, llm)
            else:
                raise ValueError(
                    f"Unsupported attribute '{attribute}'. Only 'gender' and 'race' are supported."
                )

            if terms:  # Only include prompts with detected terms
                prompts_with_terms.append(prompt)
                detected_terms_per_prompt.append(terms)

        # Print FTU status
        n_prompts_with_terms = len(prompts_with_terms)
        attribute_to_print = (
            attribute.capitalize() if attribute else "Protected attribute"
        )

        if n_prompts_with_terms == 0:
            print(
                f"LLM-based FTU check: 0 prompts contain {attribute_to_print.lower()} words"
            )
            print("FTU is satisfied.")
        else:
            print(
                f"LLM-based FTU check: {n_prompts_with_terms} prompts contain {attribute_to_print.lower()} words"
            )
            print("FTU is not satisfied.")

        return prompts_with_terms, detected_terms_per_prompt

    def _detect_race_explicit_terms(self, text: str, langchain_llm: Any) -> List[str]:
        """Detect explicit race terms only - mimics static method behavior."""
        detected_terms = []

        system_prompt = """You are an expert at detecting explicit race and ethnicity references in text.

Your task is to identify explicit race/ethnicity-related words and phrases in the given text.

Instructions:
1. Look ONLY for explicit race/ethnicity terms like: white, black, caucasian, african american, hispanic, latino, asian, native american, etc.
2. Include person descriptors when combined with race (e.g., "white male", "black woman")
3. If you find race terms, list the EXACT words or phrases from the text
4. If no explicit race terms are found, respond with "NONE"
5. Return only the specific terms, separated by commas if multiple
6. Focus on terms that would be in a traditional race word list
7. Ignore non-person contexts (e.g., "black car" should return "NONE")

Examples:
- "The caucasian male patient" → "caucasian, male"
- "She is african american" → "african american"
- "The black car" → "NONE" (not about a person)
- "Asian cuisine" → "NONE" (not about a person)
- "The hispanic woman" → "hispanic"
"""

        human_prompt = f"Text to analyze: {text}"

        from langchain_core.messages import HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = langchain_llm.invoke(messages)
            response_text = response.content.strip()

            if response_text.upper() != "NONE":
                terms = [term.strip().lower() for term in response_text.split(",")]
                detected_terms.extend(terms)
        except Exception as e:
            warnings.warn(f"LLM race detection failed: {e}")

        return detected_terms

    @staticmethod
    def _replace_race(text: str, target_race: str) -> str:
        """Replaces text with a target word"""
        seq = text.lower()
        race_replacement_mapping = {}
        for rw in (
            RACE_WORDS_REQUIRING_CONTEXT
        ):  # Include token-pairs that indicate reference to the race of a person
            for pw in PERSON_WORDS:
                key = rw + " " + pw
                race_replacement_mapping[key] = target_race + " " + pw
        for rw in RACE_WORDS_NOT_REQUIRING_CONTEXT:
            race_replacement_mapping[rw] = target_race

        for subseq in STRICT_RACE_WORDS:
            seq = seq.replace(subseq, race_replacement_mapping[subseq])
        return seq

    @staticmethod
    def _validate_attributes(
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
        custom_dict: Optional[Dict[str, str]] = None,
        for_parsing: bool = True,
    ) -> None:
        if for_parsing:
            if custom_list and attribute:
                raise ValueError("Either custom_list or attribute must be None.")
            if not (custom_list or attribute in ["race", "gender"]):
                raise ValueError(
                    "If custom_list is None, attribute must be 'race' or 'gender'."
                )
        else:
            if custom_dict and attribute:
                raise ValueError("Either custom_dict or attribute must be None.")
            if not (custom_dict or attribute in ["race", "gender"]):
                raise ValueError(
                    "If custom_dict is None, attribute must be 'race' or 'gender'."
                )
