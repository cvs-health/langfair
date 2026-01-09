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
import math
import re
from typing import Dict, List, Tuple, Union

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rich.progress import Progress

from langfair.constants.word_lists import (
    ADJECTIVE_LIST,
    GENDER_TO_WORD_LISTS,
    PROFESSION_LIST,
)
from langfair.utils.display import (
    start_progress_bar,
    stop_progress_bar,
)

# Ensuring that nltk library can access the nltk_data in 'resources' directory
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


class CooccurrenceBiasMetric:
    def __init__(
        self,
        target_category: str = "adjective",
        demographic_group_word_lists: Dict[str, List[str]] = None,
        stereotype_word_list: List[str] = None,
        beta: float = 0.95,
        how: str = "mean",
    ) -> None:
        """
        Class for computing Co-occurrence bias score. Compute co-occurrence bias scores as defined by conditional probability ratios
        based on infinite context windows. Code is based on research by Bordia & Bowman (2019):
        https://arxiv.org/abs/1904.03035. For more information on these metrics, see Bordia & Bowman (2019) :footcite:`bordia2019identifyingreducinggenderbias`.

        Parameters
        ----------
        target_category : {'adjective', 'profession'}, default = 'adjective'
            The target category used to measure the COBS score with the COBS score with default
            target word list. Not used if `stereotype_word_list` is provided.

        demographic_group_word_lists : Dict[str, List[str]], default = None
            A dictionary with values that are demographic word lists. Must have exactly two keys.
            Each value must be a list of strings. If None, default gender word lists are used.

        stereotype_word_list : List[str], default = None
            A list of target (stereotype) words for computing COBS score. If None, a default
            word list is used based on selected `target_category`. If specified, this
            parameter takes precedence over `target_category`.

        beta : float, default=0.95
            Specifies the weighting factor for infinite context window used when calculating
            co-occurrence bias score.

        how : str, default='mean'
            If defined as 'mean', evaluate method returns average COBS score. If 'word_level', the
            method returns dictinary with COBS(w) for each word 'w'.
        """
        # Specify whether target words should be adjectives or professions
        if stereotype_word_list is not None:
            assert isinstance(stereotype_word_list, list) and all(
                isinstance(elem, str) for elem in stereotype_word_list
            ), """
                If provided, `stereotype_word_list` must be list of strings
            """
            self.target_word_list = stereotype_word_list
        else:
            assert target_category in ["adjective", "profession"]
            self.target_word_list = (
                ADJECTIVE_LIST if target_category == "adjective" else PROFESSION_LIST
            )

        if demographic_group_word_lists is None:
            self.demographic_group_word_lists = GENDER_TO_WORD_LISTS
            self.group1_nouns = set(self.demographic_group_word_lists["female"])
            self.group2_nouns = set(self.demographic_group_word_lists["male"])
        else:
            assert len(demographic_group_word_lists.keys()) == 2, """
                `demographic_group_word_lists` must be a dictionary with exactly two keys
            """

            for key in demographic_group_word_lists.keys():
                val = demographic_group_word_lists[key]
                assert isinstance(val, list) and all(
                    isinstance(elem, str) for elem in val
                ), """
                    values in `demographic_group_word_lists` must be lists of strings
                """
            self.demographic_group_word_lists = demographic_group_word_lists
            self.group1_nouns, self.group2_nouns = map(
                set, demographic_group_word_lists.values()
            )

        self.protected_nouns = self.group1_nouns | self.group2_nouns
        self.beta = beta
        self.how = how
        self.name = "Cooccurrence Bias"
        self.progress_bar = None

    def evaluate(
        self,
        responses: List[str],
        show_progress_bars: bool = True,
        existing_progress_bar: Progress = None,
    ) -> Union[float, Dict[str, float]]:
        """
        Compute the relative co-occurrence rates of target words with
        protected attribute words.

        Parameters
        ----------
        responses : list of strings
            A list of generated outputs from a language model on which co-occurrence bias score
            metric will be calculated.

        show_progress_bars : bool, default=True
            If True, displays progress bars while evaluating metrics.

        existing_progress_bar : rich.progress.Progress, default=None
            If provided, the progress bar will be updated with the existing progress bar.

        Returns
        -------
        float
            Co-occurrence bias score metric

        References
        ----------
        .. footbibliography::
        """
        if show_progress_bars:
            self.progress_bar = start_progress_bar(existing_progress_bar)

        # Conduct intermediate operations before COBS calculations
        tot_co_counts, tot_cooccur, reference_words, all_words, attribute_word_lists = (
            self._prep_lists(responses)
        )

        if not all_words:
            stop_progress_bar(self.progress_bar)
            return None

        cobs_scores = {}
        cobs_scores_list = None
        for target_word in self.target_word_list:
            # skip target word if not contained in text corpus
            if target_word not in all_words:
                continue

            # Calculate COBS(w) = P(w|f)/P(w|m) for target word w
            if (tot_co_counts[target_word]["group1"] > 0) and (
                tot_co_counts[target_word]["group2"]
            ):
                group1_numerator, group2_numerator = (
                    tot_co_counts[target_word][g] / tot_cooccur[g]
                    for g in ["group1", "group2"]
                )
                group1_denominator, group2_denominator = (
                    len(attribute_word_lists[g]) / len(reference_words)
                    for g in ["group1", "group2"]
                )
                cobs_scores[target_word] = abs(
                    math.log10(
                        (group1_numerator / group1_denominator)
                        / (group2_numerator / group2_denominator)
                    )
                )
        cobs_scores_list = [float(s) for s in cobs_scores.values() if s is not None]

        if len(cobs_scores_list) == 0:
            if show_progress_bars:
                self.progress_bar.add_task(
                    "[No Progress Bar] None of the target words co-occur with both lists of attribute words. Unable to calculate COBS score."
                )
            elif not existing_progress_bar:
                print(
                    "None of the target words co-occur with both lists of attribute words. Unable to calculate COBS score."
                )
            mean_cobs_score = None
        else:
            mean_cobs_score = (
                np.mean(cobs_scores_list) if self.how == "mean" else cobs_scores
            )
        stop_progress_bar(self.progress_bar)
        return mean_cobs_score

    def _prep_lists(
        self,
        responses: List[str],
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Create lists for COBS(w) calculation from list of responses.

        Parameters
        ----------
        responses : list of strings
            A list of generated outputs from a language model on which co-occurrence bias score
            metric will be calculated.
        """
        # Tokenize sentences, get list of all words, and get set of non-protected, non-stop words
        tokenized_texts = [self._get_clean_token_list(t) for t in responses]
        all_words = list(itertools.chain(*tokenized_texts))
        reference_words = [
            word
            for word in all_words
            if word not in stop_words.union(self.protected_nouns)
        ]

        # Get list of both sets of protected attribute words contained in text corpus
        attribute_word_lists = {}
        attribute_word_lists["group1"] = [
            word for word in all_words if word in self.group1_nouns
        ]
        attribute_word_lists["group2"] = [
            word for word in all_words if word in self.group2_nouns
        ]
        if not (
            (len(attribute_word_lists["group1"]) > 0)
            and (len(attribute_word_lists["group2"]) > 0)
        ):
            if self.progress_bar:
                self.progress_bar.add_task(
                    "[No Progress Bar] The provided sentences do not contain words from both word lists. Unable to calculate Co-occurrence bias score."
                )
            else:
                print(
                    "The provided sentences do not contain words from both word lists. Unable to calculate Co-occurrence bias score."
                )
            return None, None, None, None, None
        tot_co_counts = {}
        if self.progress_bar:
            progress_bar_task = self.progress_bar.add_task(
                "Computing Co-Occurrence Bias Scores...",
                total=len(tokenized_texts),
            )
        for text in tokenized_texts:
            # Get procted attribute cooccurrence counts for current sentence
            co_counts = self._calculate_cooccurrence_scores(text, self.beta)

            # Get cumulative protected attribute cooccurrence counts
            tot_co_counts = {
                word: {
                    "group1": co_counts.get(word, {"group1": 0, "group2": 0})["group1"]
                    + tot_co_counts.get(word, {"group1": 0, "group2": 0})["group1"],
                    "group2": co_counts.get(word, {"group1": 0, "group2": 0})["group2"]
                    + tot_co_counts.get(word, {"group1": 0, "group2": 0})["group2"],
                }
                for word in set(co_counts) | set(tot_co_counts)
            }
            if self.progress_bar:
                self.progress_bar.update(progress_bar_task, advance=1)

        # Get total cooccurrence counts for all words for COBS calculation
        tot_cooccur = {"group1": 0, "group2": 0}
        for word in set(reference_words):
            for g in ["group1", "group2"]:
                tot_cooccur[g] += tot_co_counts[word][g]

        return (
            tot_co_counts,
            tot_cooccur,
            reference_words,
            all_words,
            attribute_word_lists,
        )

    def _calculate_cooccurrence_scores(
        self, response: str, beta: float
    ) -> Dict[str, float]:
        """
        Gets cooccurrences of each word in a text corpus with protected attribute words. Returns a dictionary
        indicating group word cooccurrences for each target word.
        """
        cooccurence_scores = {}
        response_words = list(enumerate(response))
        for ref_pos, ref_word in response_words:
            if ref_word not in stop_words.union(self.protected_nouns):
                if ref_word not in cooccurence_scores:
                    cooccurence_scores[ref_word] = {"group1": 0, "group2": 0}

                for attribute_pos, attribute_word in response_words:
                    token_distance = abs(ref_pos - attribute_pos)

                    if (attribute_word in self.group1_nouns) and (token_distance > 0):
                        cooccurence_scores[ref_word]["group1"] += pow(
                            beta, token_distance
                        )

                    if attribute_word in self.group2_nouns and (token_distance > 0):
                        cooccurence_scores[ref_word]["group2"] += pow(
                            beta, token_distance
                        )
        return cooccurence_scores

    def _get_clean_token_list(self, text: str) -> List[str]:
        """
        Get clean token list.
        """
        return [
            t
            for t in (self._transform_token(w) for w in word_tokenize(text))
            if t != ""
        ]

    @staticmethod
    def _transform_token(w: str) -> str:
        """
        Makes token lowercase, replaces digits with placeholder, and .
        """
        w = re.sub(r"\d+", "NUMBER", w.lower())  # Replace digits with placeholder
        w = re.sub(
            r"^[^A-Za-z<>$]+|[^A-Za-z<>$]+$", "", w
        )  # Remove unwanted characters and leading/trailing punctuation
        return w
