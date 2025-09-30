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

import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Metric(ABC):
    """
    Abstract base class of fractaion, expected maximum, and probability metrics. Serves as a
    template for creating new metric functions.
    """

    @abstractmethod
    def metric_function(self, scores: List[float], toxic_threshold: float) -> float:
        """
        Abstract method that needs to be implemented by the user when creating a new metric function.
        """
        pass

    def evaluate(
        self,
        data: Dict[str, Any],
        threshold: float,
        score_column: str = "score",
    ) -> float:
        """
        This method compute metric function for unique input prompts and return the mean value over all
        unique prompts.

        Parameters
        ----------
        data: dict
            A dictionary containing toxicity/stereotype score.

        threshold : float
            Specifies the threshold to use for toxicity/stereotype score classification.

        score_column : str, default='score'
            Name of the dictionary key that contains score.

        show_progress_bars : bool, default=True
            If True, displays progress bars while evaluating metrics.

        existing_progress_bar : rich.progress.Progress, default=None
            If provided, the progress bar will be updated with the existing progress bar.
        """
        results = []
        for prompt in set(data["prompt"]):
            score = [
                data[score_column][i]
                for i in range(len(data[score_column]))
                if data["prompt"][i] == prompt
            ]
            results.append(self.metric_function(score, threshold))
        return statistics.mean(results)
