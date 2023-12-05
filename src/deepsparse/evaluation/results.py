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

import json
from collections import OrderedDict
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
    "save_evaluations",
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
        description="Evaluation results represented in the unified, structured format"
    )
    raw: Any = Field(
        description="Evaluation results represented in the raw format "
        "(characteristic for the specific evaluation integration)"
    )

    def __str__(self):
        """
        The string representation of the Result object is
        the formatted evaluation results serialized in JSON.
        """
        return save_evaluations(self.formatted, save_format="json", save_path=None)


def save_evaluations(
    evaluations: List[Evaluation], save_format: str = "json", save_path: str = None
):
    """
    Saves a list of Evaluation objects to a file in the specified format.

    :param evaluations: List of Evaluation objects to save
    :param save_format: Format to save the evaluations in.
    :param save_path: Path to save the evaluations to.
        If None, the evaluations will not be saved.
    :return: The serialized evaluations
    """
    # serialize the evaluations
    evaluations: List[Evaluation] = prep_for_serialization(evaluations)
    # convert to ordered dicts to preserve order
    evaluations: List[OrderedDict] = evaluations_to_dicts(evaluations)
    if save_format == "json":
        return _save_to_json(evaluations, save_path)
    elif save_format == "yaml":
        return _save_to_yaml(evaluations, save_path)
    else:
        NotImplementedError("Currently only json and yaml formats are supported")


def _save_to_json(evaluations: List[OrderedDict], save_path: Optional[str]) -> str:
    data = json.dumps(evaluations, indent=4)
    if save_path:
        _save(data, save_path, expected_ext=".json")
    return data


def _save_to_yaml(evaluations: List[OrderedDict], save_path: Optional[str]) -> str:
    # required to properly process OrderedDicts
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping(
            "tag:yaml.org,2002:map", data.items()
        ),
    )
    data = yaml.dump(evaluations, default_flow_style=False)
    if save_path:
        _save(data, save_path, expected_ext=".yaml")
    return data


def evaluations_to_dicts(evaluations: List[Evaluation]):
    return [OrderedDict(**evaluation.dict()) for evaluation in evaluations]


def _save(data: str, save_path: str, expected_ext: str):
    if not save_path.endswith(expected_ext):
        raise ValueError(f"save_path must end with extension: {expected_ext}")
    with open(save_path, "w") as f:
        f.write(data)
