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
from typing import List

import numpy

from deepsparse.loggers.metric_functions.registry import (
    register as register_metric_function,
)
from deepsparse.loggers.metric_functions.utils import BatchResult


__all__ = ["mean_score", "percent_zero_labels"]


@register_metric_function(group="token_classification", identifier="pipeline_outputs")
def percent_zero_labels(
    token_classification_output: "TokenClassificationOutput",  # noqa: F821
) -> BatchResult:
    """
    Returns the percentage of zero labels in the token classification output

    :param token_classification_output: the TokenClassificationOutput object
    :return: BatchResult object, that contains the percentage of zero labels
        in for each sequence of tokens in the batch
    """
    batch_result = BatchResult()
    for prediction in token_classification_output.predictions:
        batch_result.append(_percent_zero_labels(prediction))
    return batch_result


@register_metric_function(group="token_classification", identifier="pipeline_outputs")
def mean_score(
    token_classification_output: "TokenClassificationOutput",  # noqa: F821
) -> BatchResult:
    """
    Returns the mean score of the token classification output

    :param token_classification_output: the TokenClassificationOutput object
    :return: BatchResult object, that contains the mean score for each
        sequence of tokens in the batch
    """
    batch_result = BatchResult()
    for prediction in token_classification_output.predictions:
        batch_result.append(_mean_score(prediction))
    return batch_result


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
