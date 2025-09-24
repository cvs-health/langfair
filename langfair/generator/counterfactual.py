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
from langchain_core.messages.system import SystemMessage
from nltk.tokenize import word_tokenize

# Optional imports for LLM-based FTU checking
try:
    from transformers import pipeline
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

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
        ftu_result: Optional[Dict[str, Any]] = None,
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

        ftu_result : Dict[str, Any], default=None
            Optional FTU result from check_ftu(). If provided and method is 'llm', will use 
            LLM-detected terms for more precise substitution.

        Returns
        -------
        dict
            Dictionary containing counterfactual prompts
        """
        self._validate_attributes(
            attribute=attribute, custom_dict=custom_dict, for_parsing=False
        )

        # Check if we should use LLM-detected terms
        if ftu_result and ftu_result.get("metadata", {}).get("method") == "llm":
            # Use LLM-detected terms for more precise substitution
            return self.create_prompts_from_llm_terms(
                prompts=ftu_result["data"]["prompt"],
                llm_detected_terms=ftu_result["data"]["attribute_words"],
                attribute=attribute,
                custom_dict=custom_dict,
            )

        # Fall back to traditional static method
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
            return {
                "original_prompt": [],
                "attribute_words": [],
            }
        
        if attribute == "race":
            # For race, create counterfactuals for each race group
            prompts_dict = {
                race + "_prompt": self._counterfactual_sub_race_with_terms(
                    texts=filtered_prompts, 
                    detected_terms=filtered_terms,
                    target_race=race
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
                    ref_dict=ref_dict, 
                    text=prompt,
                    detected_terms=terms
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
                    # Direct replacement for standalone race words
                    new_text = re.sub(rf'\b{re.escape(term)}\b', target_race, new_text, flags=re.IGNORECASE)
                elif any(term.startswith(rw + " ") for rw in RACE_WORDS_REQUIRING_CONTEXT):
                    # Handle race + person combinations
                    parts = term.split(" ", 1)
                    if len(parts) == 2:
                        race_part, person_part = parts
                        replacement = f"{target_race} {person_part}"
                        new_text = re.sub(rf'\b{re.escape(term)}\b', replacement, new_text, flags=re.IGNORECASE)
            new_texts.append(new_text)
        return new_texts
    
    def _sub_from_dict_with_terms(
        self, 
        ref_dict: Dict[str, List[str]], 
        text: str,
        detected_terms: List[str]
    ) -> Dict[str, str]:
        """Creates counterfactual variations using detected terms."""
        # For now, use the original method but could be enhanced to be more targeted
        return self._sub_from_dict(ref_dict=ref_dict, text=text)

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
        ftu_result: Optional[Dict[str, Any]] = None,
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

        ftu_result : Dict[str, Any], default=None
            Optional FTU result from check_ftu(). If provided and method is 'llm', will use 
            LLM-detected terms for more precise counterfactual generation.

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
        prompts_dict = self.create_prompts(
            prompts=prompts,
            attribute=attribute,
            custom_dict=custom_dict,
            ftu_result=ftu_result,
        )

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
            responses_dict[group + "_response"] = self._enforce_strings(tmp_responses)
            # stop = time.time()

        print("Responses successfully generated!")
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
            },
        }


    def check_ftu(
        self,
        prompts: List[str],
        attribute: Optional[str] = None,
        custom_list: Optional[List[str]] = None,
        subset_prompts: bool = True,
        llm: Union[bool, "BaseChatModel"] = False,
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

        llm : Union[bool, BaseChatModel], default=False
            Controls the detection method:
            - False: Uses static word lists (default, fastest)
            - True: Uses HuggingFace models (requires transformers library)
            - BaseChatModel: Uses the provided LangChain model (e.g., ChatVertexAI, ChatOpenAI)

        llm_threshold : float, default=0.3
            Confidence threshold for LLM-based classification (0.0 to 1.0). Default of 0.3 is 
            intentionally low to err on the side of caution in bias detection - false negatives 
            are more problematic than false positives in fairness contexts. Only used when llm is not False.

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
        if llm is not False:
            return self._check_ftu_llm(
                prompts=prompts,
                attribute=attribute,
                custom_list=custom_list,
                subset_prompts=subset_prompts,
                llm=llm,
                llm_threshold=llm_threshold,
            )
        else:
            return self._check_ftu_static(
                prompts=prompts,
                attribute=attribute,
                custom_list=custom_list,
                subset_prompts=subset_prompts,
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
        llm: Union[bool, "BaseChatModel"] = True,
        llm_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Internal method for LLM-based FTU checking."""
        # Determine which LLM approach to use
        use_langchain_llm = llm is not True  # If not True, it should be a LangChain model
        
        if use_langchain_llm:
            # Validate that we have a proper LangChain model
            if LANGCHAIN_AVAILABLE:
                if not isinstance(llm, BaseChatModel):
                    raise ValueError(
                        f"Provided llm must be a LangChain BaseChatModel, got {type(llm).__name__}. "
                        "Example: from langchain_google_vertexai import ChatVertexAI; llm = ChatVertexAI()"
                    )
            else:
                # Fallback check if langchain_core not available
                if not hasattr(llm, 'invoke'):
                    raise ValueError(
                        "Provided llm must be a LangChain BaseChatModel with invoke() method. "
                        "Install langchain_core for better type checking."
                    )
        else:
            # Using HuggingFace - check if transformers is available
            if not HF_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers library not available. Install with: pip install transformers torch"
                )

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
                prompt_attributes = self._check_custom_attributes_llm(prompt, custom_list, llm_threshold, llm if use_langchain_llm else None)
            else:
                # Use the existing LLM detection methods
                if use_langchain_llm:
                    prompt_attributes = self._check_attributes_with_langchain(prompt, attributes, llm_threshold, llm)
                else:
                    prompt_attributes = self._check_attributes_with_hf(prompt, attributes, "facebook/bart-large-mnli", llm_threshold)
            
            if prompt_attributes:
                detected_prompts.append(prompt)
                detected_attributes.append(prompt_attributes)
            elif not subset_prompts:
                detected_prompts.append(prompt)
                detected_attributes.append([])
        
        n_prompts_with_attributes = len([attrs for attrs in detected_attributes if attrs])
        ftu_satisfied = n_prompts_with_attributes > 0
        
        attribute_to_print = (
            "Protected attribute" if not attribute else attribute.capitalize()
        )
        ftu_text = " not " if ftu_satisfied else " "
        print(f"LLM-based FTU check: {n_prompts_with_attributes} prompts contain {attribute_to_print.lower()} words")
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
                "llm_type": "langchain" if use_langchain_llm else "huggingface",
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
        langchain_llm: Optional[Any] = None
    ) -> List[str]:
        """Check for custom attributes using LLM."""
        # For custom lists, do a simple check if any terms from the list are mentioned
        detected_terms = []
        
        if langchain_llm:
            # Use LangChain LLM to check for custom terms
            system_prompt = f"""You are an expert at detecting specific terms in text.

Your task is to identify if any of these terms appear in the given text: {', '.join(custom_list)}

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
                HumanMessage(content=human_prompt)
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
                pattern = rf'\b{re.escape(rw)}\s+{re.escape(pw)}\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_words.append(phrase)
        
        # Check words not requiring context (e.g., "caucasian", "hispanic")
        for rw in RACE_WORDS_NOT_REQUIRING_CONTEXT:
            # Use word boundaries to match complete words
            pattern = rf'\b{re.escape(rw)}\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_words.append(rw)
        
        return found_words

    
    def _check_attributes_with_hf(
        self, 
        text: str, 
        attributes: List[str], 
        model_name: str, 
        threshold: float
    ) -> List[str]:
        """Check for protected attributes using HuggingFace model and extract specific terms."""
        detected_terms = []
        
        for attribute in attributes:
            # First, check if the attribute is present
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            
            if attribute == "race":
                labels = ["mentions race or ethnicity", "does not mention race or ethnicity"]
            elif attribute == "gender":
                labels = ["mentions gender", "does not mention gender"]
            else:
                labels = [f"mentions {attribute}", f"does not mention {attribute}"]
            
            result = classifier(text, labels)
            positive_label = labels[0]
            
            if result['labels'][0] == positive_label and result['scores'][0] >= threshold:
                # If attribute detected, extract specific terms using NER or keyword extraction
                specific_terms = self._extract_specific_terms_hf(text, attribute)
                detected_terms.extend(specific_terms)
        
        return detected_terms
    
    def _extract_specific_terms_hf(self, text: str, attribute: str) -> List[str]:
        """Extract specific terms for a detected attribute using HuggingFace models."""
        detected_terms = []
        
        if attribute == "race":
            # Use a combination of NER and keyword matching for race terms
            try:
                # Try using NER to find person-related entities
                ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=-1)
                ner_results = ner_pipeline(text)
                
                # Look for race-related terms in the text using a more targeted approach
                text_lower = text.lower()
                for race_word in RACE_WORDS_NOT_REQUIRING_CONTEXT:
                    if race_word in text_lower:
                        detected_terms.append(race_word)
                
                # Check for race + person combinations
                for race_word in RACE_WORDS_REQUIRING_CONTEXT:
                    for person_word in PERSON_WORDS:
                        phrase = f"{race_word} {person_word}"
                        if phrase in text_lower:
                            detected_terms.append(phrase)
                            
            except Exception:
                # Fallback to simple keyword matching
                text_lower = text.lower()
                for race_word in RACE_WORDS_NOT_REQUIRING_CONTEXT + RACE_WORDS_REQUIRING_CONTEXT:
                    if race_word in text_lower:
                        detected_terms.append(race_word)
        
        elif attribute == "gender":
            # For gender, use token-based matching similar to the original approach
            tokens = word_tokenize(text.lower())
            for gender_word in ALL_GENDER_WORDS:
                if gender_word in tokens:
                    detected_terms.append(gender_word)
        
        return list(set(detected_terms))  # Remove duplicates
    
    def _check_attributes_with_langchain(
        self, 
        text: str, 
        attributes: List[str], 
        threshold: float,
        langchain_llm: Any
    ) -> List[str]:
        """Check for protected attributes using LangChain LLM and extract specific terms."""
        detected_terms = []
        
        for attribute in attributes:
            # Create a prompt for the LLM to both detect and extract terms
            system_prompt = f"""You are an expert at detecting protected attributes in text.

Your task is to identify specific {attribute} terms or phrases in the given text.

Instructions:
1. Look for any references to {attribute} in the text
2. If you find {attribute} references, list the EXACT words or phrases from the text
3. If no {attribute} references are found, respond with "NONE"
4. Be precise and consider context - avoid false positives from partial word matches
5. Return only the specific terms, separated by commas if multiple

Examples:
- For "The caucasian male patient" → "caucasian, male" 
- For "She is a doctor" → "she"
- For "The black car" → "NONE" (not about a person)"""
            
            human_prompt = f"Text to analyze: {text}"
            
            # Use the LangChain LLM
            from langchain_core.messages import HumanMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            try:
                response = langchain_llm.invoke(messages)
                response_text = response.content.strip()
                
                if response_text.upper() != "NONE":
                    # Parse the comma-separated terms
                    terms = [term.strip().lower() for term in response_text.split(",")]
                    detected_terms.extend(terms)
                    
            except Exception as e:
                warnings.warn(f"LLM term extraction failed for {attribute}: {e}")
        
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
