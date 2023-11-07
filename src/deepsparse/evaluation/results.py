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
from pydantic import BaseModel

from src.deepsparse.utils.data import prep_for_serialization


# TODO: Finish docstrings
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
    task: str
    dataset: Dataset
    metrics: List[Metric]
    samples: List[EvalSample]


def save_evaluation(
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
    evaluations = prep_for_serialization(evaluations)
    # convert to ordered dicts to preserve order
    evaluations = [OrderedDict(**evaluation.dict()) for evaluation in evaluations]
    if save_format == "json":
        return _save_to_json(evaluations, save_path)
    elif save_format == "yaml":
        return _save_to_yaml(evaluations, save_path)
    else:
        NotImplementedError("Currently only json and yaml formats are supported")


def _save_to_json(evaluations: List[OrderedDict], save_path: Optional[str]) -> str:
    data = json.dumps(evaluations)
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


def _save(data: str, save_path: str, expected_ext: str):
    if not save_path.endswith("expected_ext"):
        raise ValueError("save_path must end " f"with extension: {expected_ext}")
    with open(save_path, "w") as f:
        f.write(data)
