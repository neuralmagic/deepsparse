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

import pytest
from deepsparse.loggers.config import MetricFunctionConfig
from deepsparse.loggers.metric_functions.helpers.config_generation import (
    _loggers_to_config_string,
    _metric_function_config_to_string,
    _metric_functions_configs_to_string,
    _nested_dict_to_lines,
)


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
