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

from typing import List, Union, Dict

import numpy
from pydantic import BaseModel


def percent_zero_labels(
    token_classification_output: "TokenClassificationOutput"
) -> Dict[str, float]:
    """
    Returns the percentage of zero labels in the token classification output
    :param token_classification_output:
    :return:
    """
    result = {}
    for prediction_idx, prediction in enumerate(token_classification_output.predictions):
        result[str(prediction_idx)] = _percent_zero_labels(prediction)
    return result


def mean_score(token_classification_output: "TokenClassificationOutput"):
    result = {}
    for prediction_idx, prediction in enumerate(token_classification_output.predictions):
        result[str(prediction_idx)] = _mean_score(prediction)
    return result


def _mean_score(token_classification_output: List["TokenClassificationResult"]) -> float:
    return numpy.mean([result.score for result in token_classification_output])


def _percent_zero_labels(token_classification_output: List["TokenClassificationResult"]) -> float:
    label_zero = "LABEL_0"
    all_results = len(token_classification_output)
    zero_results = sum(
        1 for result in token_classification_output if result.entity == label_zero
    )
    return zero_results / all_results
