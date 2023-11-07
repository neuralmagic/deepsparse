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

import numpy as np
import yaml

import pytest
from src.deepsparse.evaluation.output_evaluation import (
    Dataset,
    EvalSample,
    Evaluation,
    Metric,
    save_evaluation,
)


@pytest.fixture()
def evaluations():
    return [
        Evaluation(
            task="task_1",
            dataset=Dataset(
                type="type_1", name="name_1", config="config_1", split="split_1"
            ),
            metrics=[Metric(name="metric_name_1", value=1.0)],
            samples=[EvalSample(input=np.array([[5]]), output=5)],
        ),
        Evaluation(
            task="task_2",
            dataset=Dataset(
                type="type_2", name="name_2", config="config_2", split="split_2"
            ),
            metrics=[
                Metric(name="metric_name_2", value=2.0),
                Metric(name="metric_name_3", value=3.0),
            ],
            samples=[
                EvalSample(input=np.array([[10.0]]), output=10.0),
                EvalSample(input=np.array([[20.0]]), output=20.0),
            ],
        ),
    ]


@pytest.fixture()
def evaluations_json():
    return """[{"task": "task_1", "dataset": {"type": "type_1", "name": "name_1", "config": "config_1", "split": "split_1"}, "metrics": [{"name": "metric_name_1", "value": 1.0}], "samples": [{"input": [[5]], "output": 5}]}, {"task": "task_2", "dataset": {"type": "type_2", "name": "name_2", "config": "config_2", "split": "split_2"}, "metrics": [{"name": "metric_name_2", "value": 2.0}, {"name": "metric_name_3", "value": 3.0}], "samples": [{"input": [[10.0]], "output": 10.0}, {"input": [[20.0]], "output": 20.0}]}]"""  # noqa: E501


@pytest.fixture()
def evaluations_yaml():
    return """- task: task_1
  dataset:
    config: config_1
    name: name_1
    split: split_1
    type: type_1
  metrics:
  - name: metric_name_1
    value: 1.0
  samples:
  - input:
    - - 5
    output: 5
- task: task_2
  dataset:
    config: config_2
    name: name_2
    split: split_2
    type: type_2
  metrics:
  - name: metric_name_2
    value: 2.0
  - name: metric_name_3
    value: 3.0
  samples:
  - input:
    - - 10.0
    output: 10.0
  - input:
    - - 20.0
    output: 20.0
"""


def test_serialize_evaluation_json(tmp_path, evaluations, evaluations_json):
    path_to_file = tmp_path / "result.json"
    evaluations_serialized = save_evaluation(
        evaluations=evaluations, format="json", save_path=path_to_file.as_posix()
    )
    with open(path_to_file.as_posix(), "r") as f:
        assert json.load(f)
    assert evaluations_serialized == evaluations_json


def test_serialize_evaluation_yaml(tmp_path, evaluations, evaluations_yaml):
    path_to_file = tmp_path / "result.yaml"
    evaluations_serialized = save_evaluation(
        evaluations=evaluations, format="yaml", save_path=path_to_file.as_posix()
    )
    with open(path_to_file.as_posix(), "r") as f:
        assert yaml.safe_load(f)
    assert evaluations_serialized == evaluations_yaml
