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
The set of the general built-in metric functions
"""
from typing import Any, List, Union

from deepsparse.loggers.metric_functions.registry import (
    register as register_metric_function,
)
from deepsparse.loggers.metric_functions.utils import BatchResult


__all__ = ["identity", "predicted_classes", "predicted_top_score"]


def identity(x: Any):
    """
    Simple identity function

    :param x: Any object
    :return: The same object
    """
    return x


@register_metric_function(
    group=[
        "image_classification",
        "sentiment_analysis",
        "zero_shot_text_classification",
        "text_classification",
    ],
    identifier="pipeline_outputs.labels",
)
def predicted_classes(
    classes: List[Union[int, str, List[int], List[str]]]
) -> BatchResult:
    """
    Returns the predicted classes from the model output
    schema in the form of a BatchResult object

    :param classes: The classes to convert to a BatchResult
    """

    if isinstance(classes[0], list):
        result = BatchResult()
        for class_ in classes:
            result.append(
                BatchResult([_check_if_convertable_to_int(value) for value in class_])
            )
        return result
    else:
        return BatchResult([_check_if_convertable_to_int(value) for value in classes])


@register_metric_function(
    group=[
        "image_classification",
        "sentiment_analysis",
        "zero_shot_text_classification",
        "text_classification",
    ],
    identifier="pipeline_outputs.scores",
)
def predicted_top_score(
    scores: List[Union[float, List[float]]]
) -> Union[float, BatchResult]:
    """
    Returns the top score from the model output
    schema in the form of a BatchResult object
    (or a single float)

    :param scores: The scores to convert to a BatchResult
    """
    if isinstance(scores[0], list):
        result = BatchResult()
        for scores_ in scores:
            result.append(max(scores_))
        return result
    else:
        return max(scores)


def _check_if_convertable_to_int(value):
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
    return value
