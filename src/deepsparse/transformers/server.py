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
Schemas for requests and responses to/from the Inference Server
Note: The Schemas are specific to the task at hand and are used by FastApi
for validation and docs generation
"""

from typing import List, Optional, Union

from pydantic import BaseModel

from deepsparse.server.config import ServeModelConfig, ServerConfig
from deepsparse.tasks import SupportedTasks
from deepsparse.transformers import Pipeline, pipeline


__all__ = [
    "create_pipeline_definitions",
    "QuestionAnsweringRequest",
    "QuestionAnsweringResponse",
    "TextClassificationRequest",
    "TextClassificationResponse",
    "TokenClassificationRequest",
    "TokenClassificationResponse",
]


def create_pipeline_definitions(model_config: ServeModelConfig):
    if SupportedTasks.nlp.question_answering.matches(model_config.task):
        request_model = QuestionAnsweringRequest
        response_model = Union[
            List[QuestionAnsweringResponse],
            QuestionAnsweringResponse,
        ]
        kwargs = {}
    elif SupportedTasks.nlp.text_classification.matches(model_config.task):
        request_model = TextClassificationRequest
        response_model = Union[
            List[TextClassificationResponse], List[List[TextClassificationResponse]]
        ]
        kwargs = {}
    elif SupportedTasks.nlp.token_classification.matches(model_config.task):
        request_model = TokenClassificationRequest
        response_model = Union[
            List[TokenClassificationResponse], List[List[TokenClassificationResponse]]
        ]
        kwargs = {}
    else:
        raise ValueError(
            f"unrecognized task given of {model_config.task} for config {model_config}"
        )

    pipeline_instance: Pipeline = pipeline(
        task=model_config.task,
        model_path=model_config.model_path,
        engine_type=model_config.engine,
        num_cores=model_config.num_cores,
        scheduler=model_config.scheduler,
        batch_size=model_config.batch_size,
        **model_config.kwargs,
    )

    return pipeline_instance, request_model, response_model, kwargs


class QuestionAnsweringRequest(BaseModel):
    """
    The request model for Question Answering Task

    :param question: Either a string or a List of string questions to answer
    :param context: Either a string or List of strings representing the context
        for each question
    """

    question: Union[List[str], str]
    context: Union[List[str], str]


class TokenClassificationRequest(BaseModel):
    """
    Schema for TokenClassificationPipeline Request

    :param inputs: A string or List of strings representing input to
        TokenClassificationPipeline task
    """

    inputs: Union[List[str], str]


class TextClassificationRequest(BaseModel):
    """
    Schema for TextClassificationPipeline Request

    :param sequences: A string or List of strings representing input to
        TextClassificationPipeline task
    """

    sequences: Union[List[str], str]


class QuestionAnsweringResponse(BaseModel):
    """
    Schema for a result from Question Answering Task

    :param score: float confidence score for prediction
    :param start: int The start index of the answer
    :param end: int The end index of the answer
    :param answer: str The predicted answer
    """

    score: float
    start: int
    end: int
    answer: str


class TokenClassificationResponse(BaseModel):
    """
    Schema for TokenClassificationPipeline Response

    :param word: The token/word classified.
    :param score: The corresponding probability for `entity`.
    :param entity: The entity predicted for that token/word (it is named
        `entity_group` when `aggregation_strategy` is not `"none"`.
    :param index: The index of the corresponding token in the sentence.
    :param start: index of the start of the corresponding entity in the sentence
        Only exists if the offsets are available within the tokenizer
    :param end: The index of the end of the corresponding entity in the sentence.
        Only exists if the offsets are available within the tokenizer
    """

    entity: str
    score: float
    index: int
    word: str
    start: Optional[int]
    end: Optional[int]


class TextClassificationResponse(BaseModel):
    """
    Schema for TextClassificationPipeline Response

    :param label: The label predicted.
    :param score: The corresponding probability.
    """

    label: str
    score: float
