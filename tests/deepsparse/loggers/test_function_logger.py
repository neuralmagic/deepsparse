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
from deepsparse.loggers.helpers import match
import math
from tests.utils import mock_engine
from deepsparse import Pipeline
from deepsparse import FunctionLogger, PythonLogger
@pytest.mark.parametrize(
    "template, identifier, expected_output",
    [
        ("engine_inputs", "image_classification.engine_inputs", (True, None)),
        ("image_classification.engine_inputs", "image_classification.engine_inputs", (True, None)),
        ("pipeline_inputs.images", "image_classification.pipeline_inputs", (True, "images")),
        ("image_classification.pipeline_inputs.images.something", "image_classification.pipeline_inputs", (True, "images.something")),

    ],
)
def test_match(template, identifier, expected_output):
    assert match(template, identifier) == expected_output
def _identity(x):
    return x
def _return_number(x):
    return 1234
@pytest.mark.parametrize(
    "identifier, function_name, frequency, function, num_iterations, expected_log_content",
     [
    ("pipeline_inputs.inputs", "identity", 2, _identity, 5, "all_your_base_are_belong_to_us"),
    ("engine_inputs", "return_one", 1, _return_number, 5, "1234")]
)
def test_function_logger(capsys, identifier, function_name, frequency, function, num_iterations, expected_log_content):
    logger = PythonLogger()
    function_logger = FunctionLogger(logger = logger, identifier = identifier, function_name = function_name, frequency = frequency, function=function)
    pipeline = Pipeline.create(
        "token_classification",
        batch_size=1,
        logger=function_logger
    )
    all_messages = []
    for iter in range(num_iterations):
        pipeline("all_your_base_are_belong_to_us")
        all_messages += [message for message in capsys.readouterr().out.split("\n") if message != ""]

    assert len(all_messages) == math.ceil(num_iterations / frequency)
    assert all(expected_log_content in message for message in all_messages)




