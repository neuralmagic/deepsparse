# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from deepsparse.loggers.metric_functions.natural_language_processing import (
    mean_score,
    percent_zero_labels,
)
from deepsparse.transformers.pipelines.token_classification import (
    TokenClassificationOutput,
    TokenClassificationResult,
)


label_0_result = TokenClassificationResult(entity="LABEL_0", score=0.3, index=0, word=0)
label_1_result = TokenClassificationResult(entity="LABEL_1", score=0.6, index=0, word=0)


@pytest.mark.parametrize(
    "schema, expected_percent",
    [
        (
            TokenClassificationOutput(
                predictions=[
                    [label_1_result, label_0_result],
                    [label_0_result, label_0_result],
                ]
            ),
            {"0": 0.5, "1": 1.0},
        )
    ],
)
def test_percent_zero_labels(schema, expected_percent):
    assert percent_zero_labels(schema) == expected_percent


@pytest.mark.parametrize(
    "schema, expected_score",
    [
        (
            TokenClassificationOutput(
                predictions=[
                    [label_1_result, label_0_result],
                    [label_0_result, label_0_result],
                ]
            ),
            {"0": 0.45, "1": 0.3},
        )
    ],
)
def test_mean_score(schema, expected_score):
    result = mean_score(schema)
    keys1, values1 = set(expected_score.keys()), set(expected_score.values())
    keys2, values2 = set(result.keys()), set(result.values())
    assert keys1 == keys2
    assert pytest.approx(list(values1)) == list(values2)
