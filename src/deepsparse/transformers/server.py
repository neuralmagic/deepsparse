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
Specs, schemas, and pipelines for use when serving transformers models
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from deepsparse.tasks import SupportedTasks
from deepsparse.transformers.pipelines import Pipeline, pipeline


try:
    from deepsparse.server.config import ServeModelConfig

    deepsparse_server_err = None
except Exception as _err:
    deepsparse_server_err = _err
    ServeModelConfig = object

try:
    from pydantic import BaseModel, Field

    pydantic_import_err = None
except Exception as _err:
    pydantic_import_err = _err
    BaseModel = object
    Field = dict


__all__ = [
    "create_pipeline_definitions",
    "QuestionAnsweringRequest",
    "QuestionAnsweringResponse",
    "TextClassificationRequest",
    "TextClassificationResponse",
    "TokenClassificationRequest",
    "TokenClassificationResponse",
]


def create_pipeline_definitions(
    model_config: ServeModelConfig,
) -> Tuple[Pipeline, Any, Any, Dict]:
    """
    Create a pipeline definition and the supporting files for a given model config
    to use for serving in the DeepSparse inference server

    :param model_config: the server model config describing the model and params
    :return: a tuple containing (the pipeline to use for inference,
        the expected request body, the expected response body,
        any additional keyword args for use with the server)
    """
    if deepsparse_server_err:
        raise deepsparse_server_err

    if pydantic_import_err:
        raise pydantic_import_err

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
        task=model_config.task.lower().replace("_", "-"),
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
    """

    question: Union[List[str], str] = Field(
        description="Either a string or a List of string questions to answer"
    )
    context: Union[List[str], str] = Field(
        description="Either a string or List of strings representing the context "
        "for each question"
    )


class TokenClassificationRequest(BaseModel):
    """
    Schema for TokenClassificationPipeline Request
    """

    inputs: Union[List[str], str] = Field(
        description="A string or List of strings representing input to"
        "TokenClassificationPipeline task"
    )


class TextClassificationRequest(BaseModel):
    """
    Schema for TextClassificationPipeline Request
    """

    sequences: Union[List[str], str] = Field(
        description="A string or List of strings representing input to"
        "TextClassificationPipeline task"
    )


class QuestionAnsweringResponse(BaseModel):
    """
    Schema for a result from Question Answering Task
    """

    score: float = Field(description="confidence score for prediction")
    start: int = Field(description="The start index of the answer")
    end: int = Field(description="The end index of the answer")
    answer: str = Field(description="The predicted answer")


class TokenClassificationResponse(BaseModel):
    """
    Schema for TokenClassificationPipeline Response
    """

    entity: str = Field(
        description="The entity predicted for that token/word (it is named"
        "`entity_group` when `aggregation_strategy` is not `none`."
    )
    score: float = Field(description="The corresponding probability for `entity`.")
    index: int = Field(
        description="The index of the corresponding token in the sentence."
    )
    word: str = Field(description="The token/word classified.")
    start: Optional[int] = Field(
        description="The index of the start of the corresponding entity in the "
        "sentence. Only exists if the offsets are available within the tokenizer"
    )
    end: Optional[int] = Field(
        description="The index of the end of the corresponding entity in the sentence. "
        "Only exists if the offsets are available within the tokenizer"
    )


class TextClassificationResponse(BaseModel):
    """
    Schema for TextClassificationPipeline Response
    """

    label: str = Field(description="The label predicted.")
    score: float = Field(description="The corresponding probability.")
