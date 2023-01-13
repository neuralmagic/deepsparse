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
from deepsparse.loggers.function_logger import _unwrap_possible_dictionary
from tests.utils import mock_engine


def _identity(x):
    return x


def _return_number(x):
    return 1234


@pytest.mark.parametrize(
    "identifier, function_name, frequency, function, num_iterations, expected_log_content",  # noqa E501
    [
        (
            "token_classification/pipeline_inputs.inputs",
            "identity",
            2,
            _identity,
            5,
            "all_your_base_are_belong_to_us",
        ),
        (
            "token_classification/engine_inputs",
            "return_one",
            1,
            _return_number,
            5,
            "1234",
        ),
    ],
)
@mock_engine(rng_seed=0)
def test_function_logger(
    engine_mock,
    capsys,
    identifier,
    function_name,
    frequency,
    function,
    num_iterations,
    expected_log_content,
):
    logger = PythonLogger()
    function_logger = FunctionLogger(
        logger=logger,
        target_identifier=identifier,
        function_name=function_name,
        frequency=frequency,
        function=function,
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


@pytest.mark.parametrize(
"value, expected_result",
[
    (10, set([("",10)])),
    ({"level_1a": 1, "level_1b": 1}, set([("level_1a", 1), ("level_1b", 1)])),
    ({"level_1a": {"level_2a": 1, "level_2b": 2}, "level_1b": 2}, set([('level_1a__level_2a', 1), ('level_1a__level_2b', 2), ('level_1b',2)])),
    ({"level_1a": {"level_2a": 1, "level_2b": 2}, "level_1b": {"level_2a": {"level_3a": 1, "level_3b": 2}}}, set([('level_1a__level_2a',1), ('level_1a__level_2b', 2), ('level_1b__level_2a__level_3a',1), ('level_1b__level_2a__level_3b',2)])),
]
)
def test_unwrap_possible_dictionary(value, expected_result):
    result = set(result for result in _unwrap_possible_dictionary(value))
    assert result == expected_result

