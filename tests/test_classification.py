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

import numpy as np

from langfair.metrics.classification import ClassificationMetrics
from langfair.metrics.classification.metrics import (
    FalseDiscoveryRateParity,
    FalseNegativeRateParity,
    FalseOmissionRateParity,
    FalsePositiveRateParity,
    PredictedPrevalenceRateParity,
)

groups = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0])
y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1])


def test_falsediscoveryrateparity():
    fdrp = FalseDiscoveryRateParity()
    x1 = fdrp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true)
    x2 = fdrp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true, ratio=True)
    assert np.round(x1, 5) == 0.25 and np.round(x2, 5) == 2.0


def test_falsenegativerateparity():
    fnrp = FalseNegativeRateParity()
    x1 = fnrp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true)
    x2 = fnrp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true, ratio=True)
    assert np.round(x1, 5) == 0.0 and np.round(x2, 5) == 1.0


def test_falseomissionrateparity():
    forp = FalseOmissionRateParity()
    x1 = forp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true)
    x2 = forp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true, ratio=True)
    assert np.round(x1, 5) == 0.1 and np.round(x2, 5) == 0.83333


def test_falsepositiverateparity():
    fprp = FalsePositiveRateParity()
    x1 = fprp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true)
    x2 = fprp.evaluate(groups=groups, y_pred=y_pred, y_true=y_true, ratio=True)
    assert np.round(x1, 5) == 0.16667 and np.round(x2, 5) == 1.5


def test_predictedprevalencerateparity():
    pprp = PredictedPrevalenceRateParity()
    x1 = pprp.evaluate(groups=groups, y_pred=y_pred)
    x2 = pprp.evaluate(groups=groups, y_pred=y_pred, ratio=True)
    assert np.round(x1, 5) == 0.05556 and np.round(x2, 5) == 1.125


def test_classificationmetrics1():
    cm = ClassificationMetrics(metric_type="all")
    x = cm.evaluate(groups=groups, y_pred=y_pred, y_true=y_true)
    assert x == {
        "FalseNegativeRateParity": 0.0,
        "FalseOmissionRateParity": 0.09999999999999998,
        "FalsePositiveRateParity": 0.16666666666666669,
        "FalseDiscoveryRateParity": 0.25,
        "PredictedPrevalenceRateParity": 0.05555555555555558,
    }


def test_classificationmetrics2():
    cm = ClassificationMetrics(metric_type="all")
    x = cm.evaluate(groups=groups, y_pred=y_pred, y_true=y_true, ratio=True)
    assert x == {
        "FalseNegativeRateParity": 1.0,
        "FalseOmissionRateParity": 0.8333333333333334,
        "FalsePositiveRateParity": 1.5,
        "FalseDiscoveryRateParity": 2.0,
        "PredictedPrevalenceRateParity": 1.125,
    }
