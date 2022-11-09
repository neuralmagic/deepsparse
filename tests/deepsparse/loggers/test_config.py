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
from deepsparse.loggers.config import MetricFunctionConfig


metric_function_config_yaml_1 = """
  func: identity
  frequency: 5
  loggers:
    - python"""

metric_function_config_yaml_2 = """
  func: numpy.max"""

metric_function_config_yaml_3 = """
  func: numpy.max
  frequency: 0"""


@pytest.mark.parametrize(
    "config_yaml, should_fail, instance_type",
    [
        (metric_function_config_yaml_1, False, MetricFunctionConfig),
        (metric_function_config_yaml_2, False, MetricFunctionConfig),
        (
            metric_function_config_yaml_3,
            True,
            MetricFunctionConfig,
        ),  # frequency cannot be zero
    ],
)
def test_function_logging_config(config_yaml, should_fail, instance_type):
    obj = yaml.safe_load(config_yaml)
    if should_fail:
        with pytest.raises(Exception):
            MetricFunctionConfig(**obj)
    else:
        assert MetricFunctionConfig(**obj)
