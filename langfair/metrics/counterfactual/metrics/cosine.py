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

from typing import Any, List, Tuple, Union

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from langfair.metrics.counterfactual.metrics.baseclass.metrics import Metric


class CosineSimilarity(Metric):
    def __init__(self, transformer: str = None, how: str = "mean") -> None:
        """Compute variations of social group substitutions of language models. This class
        enables calculation of counterfactual cosine similarity. For more information on this metric, refer to:
        https://arxiv.org/abs/2407.10853

        Parameters
        ----------
        transformer : str (HuggingFace sentence transformer), default='all-MiniLM-L6-v2'
            Specifies which huggingface sentence transformer to use when computing cosine distance. See
            https://huggingface.co/sentence-transformers?sort_models=likes#models
            for more information. The recommended sentence transformer is 'all-MiniLM-L6-v2'.

        how : {'mean','pairwise'}
            Specifies whether to return the mean cosine similarity over all counterfactual pairs or a list containing cosine
            distance for each pair.
        """
        assert how in [
            "mean",
            "pairwise",
        ], "langfair: Only 'mean' and 'pairwise' are supported."
        assert (
            transformer is not None
        ), "langfair: A HuggingFace sentence transformer must be specified when initializing the class to compute cosine similarity."
        self.name = "Cosine Similarity"
        self.how = how
        self.transformer = transformer
        self.transformer_instance = SentenceTransformer(
            f"sentence-transformers/{transformer}"
        )

    def evaluate(
        self, texts1: List[str], texts2: List[str]
    ) -> Union[float, List[float]]:
        """
        Returns mean cosine similarity between two counterfactually generated
        lists LLM outputs in vector space.

        Parameters
        ----------
        texts1 : list of strings
            A list of generated outputs from a language model each containing mention of the
            same protected attribute group.

        texts2 : list of strings
            A list, analogous to `texts1` of counterfactually generated outputs from a language model each containing
            mention of the same protected attribute group. The mentioned protected attribute group must be a different
            group within the same protected attribute as mentioned in `texts1`.

        Returns
        -------
        float
            Mean cosine similarity score for provided lists of texts.
        """
        assert len(texts1) == len(
            texts2
        ), """langfair: Lists 'texts1' and 'texts2' must be of equal length."""

        embeddings1, embeddings2 = self._get_embeddings(
            transformer=self.transformer_instance,
            texts1=list(texts1),
            texts2=list(texts2),
        )
        cosine = self._calc_cosine_sim(embeddings1, embeddings2)
        return np.mean(cosine) if self.how == "mean" else cosine

    @staticmethod
    def _get_embeddings(
        transformer: str, texts1: List[str], texts2: List[str]
    ) -> Tuple[Any, Any]:
        """
        Helper function to get embeddings
        """
        embeddings1 = transformer.encode(texts1)
        embeddings2 = transformer.encode(texts2)
        return embeddings1, embeddings2

    @staticmethod
    def _calc_cosine_sim(embeddings1: Any, embeddings2: Any) -> List[float]:
        """
        Helper function to get cosine dist
        """
        cosine_list = []
        for i in range(0, len(embeddings1)):
            cosine_i = np.dot(embeddings1[i], embeddings2[i]) / (
                norm(embeddings1[i]) * norm(embeddings2[i])
            )
            cosine_list.append(cosine_i)
        return cosine_list
