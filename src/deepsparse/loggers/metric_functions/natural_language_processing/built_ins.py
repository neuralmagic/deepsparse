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
Set of functions for logging metrics from the natural language processing pipelines
"""
from typing import List, Union

from deepsparse.loggers.metric_functions.utils import BatchResult
from deepsparse.loggers.metric_functions.registry import register

__all__ = ["string_length", "percent_unknown_tokens"]

@register(group="question_answering", identifier="pipeline_inputs.context")
@register(group="question_answering", identifier="pipeline_inputs.question")
def string_length(sequence: Union[List[str], str]) -> Union[BatchResult, int]:
    """
    Returns the length of the sequence

    :param sequence: The sequence whose length is to be returned
    :return: The length of the sequence if sequence is a string,
        else a BatchResult containing the length of each sequence
        in the batch
    """
    if isinstance(sequence, str):
        return len(sequence)
    return BatchResult([len(seq) for seq in sequence])


def percent_unknown_tokens():
    raise NotImplementedError()
