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
import math

import pytest
from deepsparse import FunctionLogger, Pipeline, PythonLogger
from deepsparse.loggers.config import MetricFunctionConfig, TargetLoggingConfig


METRIC_FUNCTION_CONFIG_1 = MetricFunctionConfig(func="builtins:identity", frequency=2)
METRIC_FUNCTION_CONFIG_2 = MetricFunctionConfig(
    func="tests/test_data/metric_functions.py:return_number",
    frequency=1,
)


@pytest.mark.parametrize(
    "target_logging_configs, num_iterations, expected_log_content, frequency",  # noqa E501
    [
        (
            [
                TargetLoggingConfig(
                    target="token_classification.pipeline_inputs.inputs",
                    functions=[METRIC_FUNCTION_CONFIG_1],
                )
            ],
            5,
            "all_your_base_are_belong_to_us",
            2,
        ),
        (
            [
                TargetLoggingConfig(
                    target="token_classification.engine_inputs",
                    functions=[METRIC_FUNCTION_CONFIG_2],
                )
            ],
            5,
            "1234",
            1,
        ),
    ],
)
def test_function_logger(
    capsys, target_logging_configs, num_iterations, expected_log_content, frequency
):
    logger = PythonLogger()
    function_logger = FunctionLogger(
        logger=logger, target_logging_configs=target_logging_configs
    )
    pipeline = Pipeline.create(
        "token_classification", batch_size=1, logger=function_logger
    )
    all_messages = []
    for iter in range(num_iterations):
        pipeline("all_your_base_are_belong_to_us")
        all_messages += [
            message for message in capsys.readouterr().out.split("\n") if message != ""
        ]

    assert len(all_messages) == math.ceil(num_iterations / frequency)
    assert all(expected_log_content in message for message in all_messages)
