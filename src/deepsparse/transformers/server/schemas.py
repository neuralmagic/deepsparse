"""
Schemas for requests and responses to/from the Inference Server
Note: The Schemas are specific to the task at hand and are used by FastApi
for validation and docs generation
"""
from typing import List, Union, Optional

from pydantic import BaseModel

__all__ = [
    'response_models',
    'request_models',
    'TaskRequestModel',
    'TaskResponseModel',
]


################################################################################
### Request Schemas
################################################################################

class TaskRequestModel(BaseModel):
    """
    Base class for Task requests
    """
    pass


class QuestionAnsweringRequest(TaskRequestModel):
    """
    The request model for Question Answering Task

    :param question: Either a string or a List of string questions to answer
    :param context: Either a string or List of strings representing the context
        for each question
    """
    question: Union[List[str], str]
    context: Union[List[str], str]


class TokenClassificationRequest(TaskRequestModel):
    """
    Schema for TokenClassificationPipeline Request

    :param inputs: A string or List of strings representing input to
        TokenClassificationPipeline task
    """
    inputs: Union[List[str], str]


class TextClassificationRequest(TaskRequestModel):
    """
    Schema for TextClassificationPipeline Request

    :param sequences: A string or List of strings representing input to
        TextClassificationPipeline task
    """
    sequences: Union[List[str], str]


################################################################################
### Response Schemas
################################################################################

class TaskResponseModel(BaseModel):
    """
    Base class for Task responses
    """
    pass


class QuestionAnsweringResponse(TaskResponseModel):
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


class TokenClassificationResponse(TaskResponseModel):
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


class TextClassificationResponse(TaskResponseModel):
    """
    Schema for TextClassificationPipeline Response

    :param label: The label predicted.
    :param score: The corresponding probability.
    """
    label: str
    score: float


################################################################################
### Model Registries
################################################################################

request_models = {
    'question-answering': QuestionAnsweringRequest,
    'sentiment-analysis': TextClassificationRequest,
    'ner': TokenClassificationRequest,
    'text-classification': TextClassificationRequest,
    'token-classification': TokenClassificationRequest,
}

response_models = {
    'question-answering': Union[
        List[QuestionAnsweringResponse],
        QuestionAnsweringResponse,
    ],
    'sentiment-analysis': Union[
        List[TextClassificationResponse],
        List[List[TextClassificationResponse]],
    ],
    'ner': Union[
        List[TokenClassificationResponse],
        List[List[TokenClassificationResponse]]
    ],
    'text-classification': Union[
        List[TextClassificationResponse],
        List[List[TextClassificationResponse]]
    ],
    'token-classification': Union[
        List[TokenClassificationResponse],
        List[List[TokenClassificationResponse]]
    ],
}
