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

import yaml

import pytest
from deepsparse.loggers.config import MetricFunctionConfig, TargetLoggingConfig


metric_function_config_yaml = """
  func: builtins:identity
  frequency: 50"""

target_logging_config_yaml = """
    target: pipeline_outputs
    functions:
        - func: builtins:identity
        - func: np.max
          frequency: 50"""


@pytest.mark.parametrize(
    "string_config_yaml, base_model",
    [
        (metric_function_config_yaml, MetricFunctionConfig),
        (target_logging_config_yaml, TargetLoggingConfig),
    ],
)
def test_function_logging_config(string_config_yaml, base_model):
    obj = yaml.safe_load(string_config_yaml)
    assert base_model(**obj)
