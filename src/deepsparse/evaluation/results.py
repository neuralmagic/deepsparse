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

from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field

from src.deepsparse.utils.data import prep_for_serialization


__all__ = [
    "Metric",
    "Dataset",
    "EvalSample",
    "Evaluation",
    "Result",
    "save_result",
]


class Metric(BaseModel):
    name: str = Field(description="Name of the metric")
    value: float = Field(description="Value of the metric")


class Dataset(BaseModel):
    type: Optional[str] = Field(description="Type of dataset")
    name: str = Field(description="Name of the dataset")
    config: Any = Field(description="Configuration for the dataset")
    split: Optional[str] = Field(description="Split of the dataset")


class EvalSample(BaseModel):
    input: Any = Field(description="Sample input to the model")
    output: Any = Field(description="Sample output from the model")


class Evaluation(BaseModel):
    task: str = Field(
        description="Name of the evaluation integration "
        "that the evaluation was performed on"
    )
    dataset: Dataset = Field(description="Dataset that the evaluation was performed on")
    metrics: List[Metric] = Field(description="List of metrics for the evaluation")
    samples: Optional[List[EvalSample]] = Field(
        description="List of samples for the evaluation"
    )


class Result(BaseModel):
    formatted: List[Evaluation] = Field(
        description="Evaluation result represented in the unified, structured format"
    )
    raw: Any = Field(
        description="Evaluation result represented in the raw format "
        "(characteristic for the specific evaluation integration)"
    )


def save_result(
    result: Result,
    save_path: str,
    save_format: str = "json",
):
    """
    Saves a list of Evaluation objects to a file in the specified format.

    :param result: Result object to save
    :param save_format: Format to save the evaluations in.
    :param save_path: Path to save the evaluations to.
    :return: The serialized evaluations
    """
    # prepare the Result object for serialization
    result: Result = prep_for_serialization(result)
    if save_format == "json":
        _save_to_json(result, save_path)
    elif save_format == "yaml":
        _save_to_yaml(result, save_path)
    else:
        NotImplementedError("Currently only json and yaml formats are supported")


def _save_to_json(result: Result, save_path: str):
    _save(result.json(), save_path, expected_ext=".json")


def _save_to_yaml(result: Result, save_path: str):
    _save(yaml.dump(result.dict()), save_path, expected_ext=".yaml")


def _save(data: str, save_path: str, expected_ext: str):
    if not save_path.endswith(expected_ext):
        raise ValueError(f"save_path must end with extension: {expected_ext}")
    with open(save_path, "w") as f:
        f.write(data)
