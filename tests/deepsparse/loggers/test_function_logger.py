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
from deepsparse import FunctionLogger, Pipeline, PythonLogger
from tests.utils import mock_engine


CONFIG_1 = {
    "pipeline_inputs": [
        {"function": "identity", "target_logger": "python", "frequency": 3}
    ]
}

CONFIG_2 = {
    "pipeline_inputs": [
        {"function": "identity", "frequency": 3},
        {"function": "identity", "frequency": 5},
    ]
}

CONFIG_3 = {
    "pipeline_inputs": [
        {"function": "identity", "target_logger": "python", "frequency": 3},
        {"function": "identity", "frequency": 5},
    ],
    "pipeline_outputs": [{"function": "identity", "frequency": 4}],
}
"""
if for some target (e.g. "pipeline_inputs") we have
"frequency" = 3 and
"num_iterations" = 14
logging will occur on iterations
0, 3, 6, 9, 12 -> and thus 5 logs collected
"""


@pytest.mark.parametrize(
    "config,num_iterations,expected_logs_count",
    [
        (CONFIG_1, 14, {"pipeline_inputs": 5}),
        (CONFIG_2, 14, {"pipeline_inputs": 8}),
        (CONFIG_3, 14, {"pipeline_inputs": 8, "pipeline_outputs": 4}),
    ],
)
@mock_engine(rng_seed=0)
def test_function_logger(engine, config, num_iterations, expected_logs_count, capsys):

    pipeline = Pipeline.create(
        "token_classification",
        batch_size=1,
        logger=FunctionLogger(logger=PythonLogger(), config=config),
    )
    all_messages = []
    for iter in range(num_iterations):
        pipeline("all_your_base_are_belong_to_us")
        all_messages += [message for message in capsys.readouterr().out.split("\n")]
    for target_name, logs_count in expected_logs_count.items():
        assert (
            len(
                [
                    m
                    for m in all_messages
                    if (target_name in m) and ("Category: data" in m)
                ]
            )
            == logs_count
        )
    assert (
        len([m for m in all_messages if "Category: system" in m]) == num_iterations * 4
    )
