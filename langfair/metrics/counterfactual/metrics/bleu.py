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

from typing import List, Union

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rich.progress import Progress

from langfair.metrics.counterfactual.metrics.baseclass.metrics import Metric
from langfair.utils.display import start_progress_bar, stop_progress_bar


class BleuSimilarity(Metric):
    def __init__(self, how: str = "mean") -> None:
        """Compute variations of social group substitutions of language models. This class
        enables calculation of counterfactual BLEU. For more information on this metric, refer to:
        https://arxiv.org/abs/2407.10853

        Parameters
        ----------
        how : {'mean','pairwise'}
            Specifies whether to return the mean bleu similarity over all counterfactual pairs or a list containing bleu
            distance for each pair.
        """
        assert how in [
            "mean",
            "pairwise",
        ], "langfair: Only 'mean' and 'pairwise' are supported."
        self.name = "Bleu Similarity"
        self.how = how
        self.progress_bar = None

        try:
            word_tokenize("Check if this function can access the required corpus")
        except LookupError:
            nltk.download("punkt_tab")

    def evaluate(
        self,
        texts1: List[str],
        texts2: List[str],
        show_progress_bars: bool = True,
        existing_progress_bar: Progress = None,
    ) -> Union[float, List[float]]:
        """
        Returns mean BLEU score between two lists of generated outputs.

        Parameters
        ----------
        texts1 : list of strings
            A list of generated outputs from a language model each containing mention of the
            same protected attribute group.

        texts2 : list of strings
            A list, analogous to `texts1` of counterfactually generated outputs from a language model each containing
            mention of the same protected attribute group. The mentioned protected attribute group must be a different
            group within the same protected attribute as mentioned in `texts1`.

        show_progress_bars : bool, default=True
            If True, displays progress bars while evaluating metrics.

        existing_progress_bar : rich.progress.Progress, default=None
            If provided, the progress bar will be updated with the existing progress bar.

        Returns
        -------
        float
            Mean BLEU score for provided lists of texts.
        """
        assert len(texts1) == len(texts2), (
            """Lists 'texts1' and 'texts2' must be of equal length."""
        )
        if show_progress_bars:
            self.progress_bar = start_progress_bar(existing_progress_bar)
            self.progress_bar_task = self.progress_bar.add_task(
                "Computing Counterfactual BLEU scores...",
                total=len(texts1),
            )
        bleu_scores = []
        for t1, t2 in zip(texts1, texts2):
            score = self._calc_bleu(t1, t2)
            bleu_scores.append(score)
            if self.progress_bar:
                self.progress_bar.update(self.progress_bar_task, advance=1)
        stop_progress_bar(self.progress_bar)
        return np.mean(bleu_scores) if self.how == "mean" else bleu_scores

    @staticmethod
    def _calc_bleu(text1: str, text2: str) -> float:
        """
        Helper function to calculate BLEU score between two sets of tokens
        """
        chencherry = SmoothingFunction()
        tokens1, tokens2 = word_tokenize(text1.lower()), word_tokenize(text2.lower())
        return min(
            sentence_bleu([tokens1], tokens2, smoothing_function=chencherry.method1),
            sentence_bleu([tokens2], tokens1, smoothing_function=chencherry.method1),
        )
