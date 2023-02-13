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

# turning off the following rules for this file:
# flake8: noqa

import numpy as np

import pytest
from deepsparse import (
    AsyncLogger,
    FunctionLogger,
    MultiLogger,
    PrometheusLogger,
    PythonLogger,
)
from tests.helpers import find_free_port


expected_str_1 = """AsyncLogger:
  MultiLogger:
    FunctionLogger:
      target identifier: some_identifier
      function name: mean
      frequency: 1
      target_logger:
        MultiLogger:
          PythonLogger
          PythonLogger
    FunctionLogger:
      target identifier: some_identifier_2
      function name: mean
      frequency: 1
      target_logger:
        MultiLogger:
          PythonLogger
          PrometheusLogger:
            port: {port}
    FunctionLogger:
      target identifier: some_identifier_3
      function name: mean
      frequency: 2
      target_logger:
        PythonLogger"""

port_1 = find_free_port()
logger_1 = AsyncLogger(
    MultiLogger(
        [
            FunctionLogger(
                logger=MultiLogger([PythonLogger(), PythonLogger()]),
                target_identifier="some_identifier",
                function=np.mean,
            ),
            FunctionLogger(
                logger=MultiLogger([PythonLogger(), PrometheusLogger(port=port_1)]),
                target_identifier="some_identifier_2",
                function=np.mean,
            ),
            FunctionLogger(
                logger=PythonLogger(),
                target_identifier="some_identifier_3",
                function=np.mean,
                frequency=2,
            ),
        ]
    )
)

expected_str_2 = """FunctionLogger:
  target identifier: some_identifier
  function name: mean
  frequency: 1
  target_logger:
    MultiLogger:
      PythonLogger
      PythonLogger
      PythonLogger
      PythonLogger
      PythonLogger
  [...]"""

port_2 = None
logger_2 = FunctionLogger(
    logger=MultiLogger([PythonLogger()] * 20),
    target_identifier="some_identifier",
    function=np.mean,
)


@pytest.mark.parametrize(
    "expected_str, logger, port",
    [(expected_str_1, logger_1, port_1), (expected_str_2, logger_2, port_2)],
)
def test_str(expected_str, logger, port):
    """
    Test the __str__ method of the BaseLogger classes
    """
    assert str(logger) == expected_str.format(port=port) if port else expected_str
