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


import os

import yaml

import pytest
from deepsparse.loggers.config import MetricFunctionConfig, PipelineLoggingConfig
from deepsparse.loggers.metric_functions.helpers.config_generation import (
    _loggers_to_config_string,
    _metric_function_config_to_string,
    _metric_functions_configs_to_string,
    _nested_dict_to_lines,
    data_logging_config_from_predefined,
)
from deepsparse.loggers.metric_functions.registry import DATA_LOGGING_REGISTRY


DATA_LOGGING_REGISTRY_W_DUMMY_GROUP = DATA_LOGGING_REGISTRY.copy()
DATA_LOGGING_REGISTRY_W_DUMMY_GROUP.update(
    {"dummy_group": {"dummy_target": ["dummy_func_1", "dummy_func_2"]}}
)

dummy_logger_config = {
    "some_logger": {"arg_1": "argument_1", "arg_2": None},
    "some_other_logger": {"arg_3": 5.6, "arg_4": 10},
}
result_1 = """python:

pipeline_outputs.labels:
  - func: predicted_classes
    frequency: 3
  - func: predicted_top_score
    frequency: 3
pipeline_inputs.images:
  - func: image_shape
    frequency: 3
  - func: mean_pixels_per_channel
    frequency: 3
  - func: std_pixels_per_channel
    frequency: 3
  - func: fraction_zeros
    frequency: 3"""

result_2 = """some_logger:
  arg_1: argument_1
  arg_2: None
some_other_logger:
  arg_3: 5.6
  arg_4: 10

pipeline_outputs.labels:
  - func: predicted_classes
    frequency: 3
  - func: predicted_top_score
    frequency: 3
pipeline_inputs.images:
  - func: image_shape
    frequency: 3
  - func: mean_pixels_per_channel
    frequency: 3
  - func: std_pixels_per_channel
    frequency: 3
  - func: fraction_zeros
    frequency: 3"""

result_3 = """some_logger:
  arg_1: argument_1
  arg_2: None
some_other_logger:
  arg_3: 5.6
  arg_4: 10

pipeline_outputs.labels:
  - func: predicted_classes
    frequency: 10
  - func: predicted_top_score
    frequency: 10
pipeline_inputs.images:
  - func: image_shape
    frequency: 10
  - func: mean_pixels_per_channel
    frequency: 10
  - func: std_pixels_per_channel
    frequency: 10
  - func: fraction_zeros
    frequency: 10
dummy_target:
  - func: dummy_func_1
    frequency: 10
  - func: dummy_func_2
    frequency: 10"""


@pytest.mark.parametrize(
    "group_names, frequency, loggers, save_dir, registry, expected_result",
    [
        ("image_classification", 3, None, True, DATA_LOGGING_REGISTRY, result_1),
        (
            "image_classification",
            3,
            dummy_logger_config,
            False,
            DATA_LOGGING_REGISTRY,
            result_2,
        ),
        (
            ["image_classification", "dummy_group"],
            10,
            dummy_logger_config,
            True,
            DATA_LOGGING_REGISTRY_W_DUMMY_GROUP,
            result_3,
        ),
    ],
)
def test_data_logging_config_from_predefined(
    tmp_path, group_names, frequency, loggers, save_dir, registry, expected_result
):
    tmp_path.mkdir(exist_ok=True)

    string_result = data_logging_config_from_predefined(
        group_names=group_names,
        frequency=frequency,
        loggers=loggers,
        save_dir=tmp_path,
        registry=registry,
    )

    assert string_result == expected_result
    assert PipelineLoggingConfig(**yaml.safe_load(string_result))

    if save_dir:
        with open(os.path.join(tmp_path, "data_logging_config.yaml"), "r") as stream:
            string_result_saved = yaml.safe_load(stream)
        assert string_result_saved == yaml.safe_load(expected_result)


result_1 = """logger_2:
  arg_1: 1
  arg_2:
    arg_3: 3
    arg_4: 4"""


@pytest.mark.parametrize(
    "loggers, expected_result",
    [
        ({"logger_2": {"arg_1": 1, "arg_2": {"arg_3": 3, "arg_4": 4}}}, result_1),
        ({"logger_1": {}}, "logger_1:"),
    ],
)
def test_loggers_to_config_string(loggers, expected_result):
    string_result = _loggers_to_config_string(loggers)
    assert string_result == expected_result


data_logging_config = {
    "target_1": [
        MetricFunctionConfig(func="some_func_1", frequency=1),
        MetricFunctionConfig(func="some_func_2", frequency=2),
        MetricFunctionConfig(func="some_func_3", frequency=5),
    ],
    "target_2": [
        MetricFunctionConfig(
            func="some_func_4",
            frequency=1,
            target_loggers=["logger_1", "logger_2", "logger_3"],
        ),
        MetricFunctionConfig(func="some_func_5", frequency=2),
        MetricFunctionConfig(func="some_func_6", frequency=5),
    ],
}
result = """target_1:
  - func: some_func_1
    frequency: 1
  - func: some_func_2
    frequency: 2
  - func: some_func_3
    frequency: 5
target_2:
  - func: some_func_4
    frequency: 1
    target_loggers:
      - logger_1
      - logger_2
      - logger_3
  - func: some_func_5
    frequency: 2
  - func: some_func_6
    frequency: 5"""


@pytest.mark.parametrize(
    "data_logging_config, expected_result",
    [
        (data_logging_config, result),
    ],
)
def test_nested_dict_to_lines(data_logging_config, expected_result):
    string_result = ("\n").join(_nested_dict_to_lines(data_logging_config))
    assert string_result == expected_result


result_1 = """- func: some_func_1
  frequency: 1"""

result_2 = """- func: some_func_1
  frequency: 1
- func: some_func_2
  frequency: 2
  target_loggers:
    - logger_1
    - logger_2
- func: some_func_3
  frequency: 5"""


@pytest.mark.parametrize(
    "list_metric_function_configs, expected_result",
    [
        ([MetricFunctionConfig(func="some_func_1", frequency=1)], result_1),
        (
            [
                MetricFunctionConfig(func="some_func_1", frequency=1),
                MetricFunctionConfig(
                    func="some_func_2",
                    frequency=2,
                    target_loggers=["logger_1", "logger_2"],
                ),
                MetricFunctionConfig(func="some_func_3", frequency=5),
            ],
            result_2,
        ),
    ],
)
def test_metric_functions_configs_to_string(
    list_metric_function_configs, expected_result
):
    string_result = _metric_functions_configs_to_string(list_metric_function_configs)
    assert string_result == expected_result


result_1 = """func: some_func_1
frequency: 1
target_loggers:
  - logger_1
  - logger_2"""

result_2 = """func: some_func_2
frequency: 2"""


@pytest.mark.parametrize(
    "metric_function_config, expected_result",
    [
        (
            MetricFunctionConfig(
                func="some_func_1", frequency=1, target_loggers=["logger_1", "logger_2"]
            ),
            result_1,
        ),
        (MetricFunctionConfig(func="some_func_2", frequency=2), result_2),
    ],
)
def test_metric_function_config_to_string(metric_function_config, expected_result):

    string_result = _metric_function_config_to_string(metric_function_config)
    assert string_result == expected_result
