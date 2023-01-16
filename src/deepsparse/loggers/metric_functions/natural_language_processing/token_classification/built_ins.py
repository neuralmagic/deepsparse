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
"""
Set of functions for logging metrics from the token classification pipeline
"""
from typing import Dict, List

import numpy


__all__ = ["mean_score", "percent_zero_labels"]


def percent_zero_labels(
    token_classification_output: "TokenClassificationOutput",  # noqa: F821
) -> Dict[str, float]:
    """
    Returns the percentage of zero labels in the token classification output

    :param token_classification_output: the TokenClassificationOutput object
    :return: A dictionary where the key is the token sequence index and the
        value is the percentage of zero labels in the sequence of tokens
    """
    result = {}
    for prediction_idx, prediction in enumerate(
        token_classification_output.predictions
    ):
        result[str(prediction_idx)] = _percent_zero_labels(prediction)
    return result


def mean_score(
    token_classification_output: "TokenClassificationOutput",  # noqa: F821
) -> Dict[str, float]:
    """
    Returns the mean score of the token classification output

    :param token_classification_output: the TokenClassificationOutput object
    :return: A dictionary where the key is the token sequence index and the
        value is the mean score of the sequence of tokens
    """
    result = {}
    for prediction_idx, prediction in enumerate(
        token_classification_output.predictions
    ):
        result[str(prediction_idx)] = _mean_score(prediction)
    return result


def _mean_score(
    token_classification_output: List["TokenClassificationResult"],  # noqa: F821
) -> float:
    return numpy.mean([result.score for result in token_classification_output])


def _percent_zero_labels(
    token_classification_output: List["TokenClassificationResult"],  # noqa: F821
) -> float:
    label_zero = "LABEL_0"
    all_results = len(token_classification_output)
    zero_results = sum(
        1 for result in token_classification_output if result.entity == label_zero
    )
    return zero_results / all_results
