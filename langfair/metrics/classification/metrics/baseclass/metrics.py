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

from abc import ABC, abstractmethod
from typing import List, Optional

from numpy.typing import ArrayLike


class Metric(ABC):
    """
    Abstract base class of all recommendation metrics. Serves as a template for creating new metric
    functions.
    """

    @abstractmethod
    def evaluate(
        self,
        groups: ArrayLike,
        y_pred: ArrayLike,
        y_true: Optional[ArrayLike] = None,
        ratio: bool = False,
    ) -> float:
        """
        Abstract method that needs to be implemented by the user when creating a new metric function.
        """
        pass

    @staticmethod
    def binary_confusion_matrix(y_true, y_pred) -> List[List[float]]:
        """
        Method for computing binary confusion matrix

        Parameters
        ----------
        y_true : Array-like
            Binary labels (ground truth values)

        y_pred : Array-like
            Binary model predictions

        Returns
        -------
        List[List[float]]
            2x2 confusion matrix

        """
        cm = [[0, 0], [0, 0]]
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                if y_pred[i] == 0:
                    cm[0][0] += 1
                else:
                    cm[1][1] += 1
            else:
                if y_pred[i] == 0:
                    cm[1][0] += 1
                else:
                    cm[0][1] += 1
        return cm
