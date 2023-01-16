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
Set of functions for logging metrics from the question answering pipeline
"""

from deepsparse.loggers.metric_functions.natural_language_processing import (
    sequence_length,
)


__all__ = ["answer_found", "answer_length", "answer_score"]


def answer_found(qa_output: "QuestionAnsweringOutput") -> bool:  # noqa: F821
    raise NotImplementedError()


def answer_length(qa_output: "QuestionAnsweringOutput") -> int:  # noqa: F821
    """
    Returns the length of the answer given the QuestionAnsweringOutput

    :param qa_output: The output schema of the question answering pipeline
    :return: The length of the answer
    """
    return sequence_length(qa_output.answer)


def answer_score(qa_output: "QuestionAnsweringOutput") -> float:  # noqa: F821
    """
    Returns the score of the answer given the QuestionAnsweringOutput

    :param qa_output: The output schema of the question answering pipeline
    :return: The score of the answer
    """
    return qa_output.score
