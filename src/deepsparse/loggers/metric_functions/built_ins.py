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
from deepsparse.loggers.metric_functions.utils import BatchResult
from deepsparse.loggers.metric_functions.registry import register

__all__ = ["identity", "predicted_classes"]


def identity(x: Any):
    """
    Simple identity function

    :param x: Any object
    :return: The same object
    """
    return x

@register(group="image_classification", identifier = "pipeline_outputs.labels")
def predicted_classes(batch_classes: List[Union[int, str, List[int], List[str]]]) -> BatchResult:
    """
    Some docstring
    """
    result = BatchResult()
    for class_ in batch_classes:
        if isinstance(class_, list):
            class_ = BatchResult(class_)
        result.append(class_)
    return result

@register(group="image_classification", identifier = "pipeline_outputs.labels")
def predicted_top_score(batch_scores: List[Union[float, List[float]]]) -> BatchResult:
    """
    Some docstring
    """
    result = BatchResult()
    for score in batch_scores:
        if isinstance(score, list):
            score = max(score)
        result.append(score)
    return result



