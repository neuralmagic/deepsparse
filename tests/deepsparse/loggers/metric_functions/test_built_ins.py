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
from deepsparse.loggers.metric_functions import predicted_classes, predicted_top_score
from deepsparse.loggers.metric_functions.utils import BatchResult


@pytest.mark.parametrize(
    "classes, expected_result",
    [
        ([1, 2, 3, 4], BatchResult([1, 2, 3, 4])),
        (["1", "2", "3", "4"], BatchResult([1, 2, 3, 4])),
        (
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            BatchResult([BatchResult([1, 2, 3, 4]), BatchResult([5, 6, 7, 8])]),
        ),
        (
            [["1", "2", "3", "4"], [5, 6, 7, 8]],
            BatchResult([BatchResult([1, 2, 3, 4]), BatchResult([5, 6, 7, 8])]),
        ),
    ],
)
def test_predicted_classes(classes, expected_result):
    assert predicted_classes(classes) == expected_result


@pytest.mark.parametrize(
    "batch_scores, expected_result",
    [
        ([[0.7, 0.8, 0.1], [0.6, 0.4, 0.1]], BatchResult([0.8, 0.6])),
        ([0.7, 0.8, 0.1], BatchResult([0.8, 0.6])),
    ],
)
def test_predicted_top_score(batch_scores, expected_result):
    assert predicted_top_score(batch_scores) == expected_result
