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

from typing import Dict, List, Union

from pydantic import BaseModel


__all__ = ["sequence_length"]


def sequence_length(sequence: Union[List[str], str]) -> Union[Dict[str, int], int]:
    """
    Returns the length of the sequence

    :param sequence: The sequence whose length is to be returned
    :return: The length of the sequence
    """
    if isinstance(sequence, str):
        return len(sequence)
    return {str(string_id): len(string) for string_id, string in enumerate(sequence)}


def percent_unknown_tokens():
    pass





def answer_score(qa_output: BaseModel) -> float:
    return qa_output.score


def percent_zero_labels(
    token_classification_output: List[Union[BaseModel, List[BaseModel]]]
):
    if isinstance(token_classification_output[0], BaseModel):
        return _percent_zero_labels(token_classification_output)
    return {
        str(result_id): _percent_zero_labels(result)
        for result_id, result in enumerate(token_classification_output)
    }


def mean_score(token_classification_output: List[Union[BaseModel, List[BaseModel]]]):
    if isinstance(token_classification_output[0], BaseModel):
        return _mean_score(token_classification_output)
    return {
        str(result_id): _mean_score(result)
        for result_id, result in enumerate(token_classification_output)
    }


def _mean_score(token_classification_output: List[BaseModel]) -> float:
    return numpy.mean([result.score for result in token_classification_output])


def _percent_zero_labels(token_classification_output: List[BaseModel]) -> float:
    label_zero = "LABEL_0"
    all_results = len(token_classification_output)
    zero_results = sum(
        1 for result in token_classification_output if result.entity == label_zero
    )
    return zero_results / all_results
