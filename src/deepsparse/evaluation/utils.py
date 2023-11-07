from pydantic import BaseModel
from typing import Any, List

class Metric(BaseModel):
    name: str
    value: float


class Dataset(BaseModel):
    type: str
    name: str
    config: str
    split: str


class EvalSample(BaseModel):
    input: Any
    output: Any


class Evaluation(BaseModel):
    # TODO: How to handle serialization of the
    # data structure (to yaml and json)
    task: str
    dataset: Dataset
    metrics: List[Metric]
    samples: List[EvalSample]

def serialize_evaluation(evaluation: Evaluation, serialize_with: str = "json"):
    if serialize_with == "json":
        return serialize_evaluation_json(evaluation)
    elif serialize_with == "yaml":
        return serialize_evaluation_json(evaluation)
    else:
        NotImplementedError()

def evaluations_to_dictionary(evaluations: List[Evaluation]):
    return [evaluation.dict() for evaluation in evaluations]