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
from deepsparse import FunctionLogger, Pipeline, PythonLogger
from deepsparse.loggers.config import PipelineLoggingConfig, TargetLoggingConfig
from tests.utils import mock_engine


CONFIG_1 = {
    "target": "pipeline_inputs",
    "mappings": [{"func": "builtins:identity", "frequency": 3}],
}

CONFIG_2 = {
    "target": "pipeline_inputs",
    "mappings": [
        {"func": "builtins:identity", "frequency": 3},
        {
            "func": "tests/deepsparse/loggers/test_data/metric_functions.py:user_defined_identity",  # noqa E501
            "frequency": 5,
        },
    ],
}

CONFIG_3 = [
    {
        "target": "pipeline_inputs",
        "mappings": [
            {"func": "builtins:identity", "frequency": 3},
            {
                "func": "tests/deepsparse/loggers/test_data/metric_functions.py:user_defined_identity",  # noqa E501
                "frequency": 5,
            },
        ],
    },
    {
        "target": "pipeline_outputs",
        "mappings": [{"func": "builtins:identity", "frequency": 4}],
    },
]

CONFIG_4 = "tests/deepsparse/loggers/test_data/function_config.yaml"

CONFIG_5 = """ 
  name: token_classification
  targets:
    - target: pipeline_inputs
      mappings:
        - func: builtins:identity
          frequency: 3
        - func: tests/deepsparse/loggers/test_data/metric_functions.py:user_defined_identity 
          frequency: 5
    - target: pipeline_outputs
      mappings:
        - func: builtins:identity
          frequency: 4"""  # noqa E501


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
        # config from TargetLoggingConfig
        (TargetLoggingConfig(**CONFIG_1), 14, {"pipeline_inputs": 5}),
        # config from a dictionary
        (CONFIG_2, 14, {"pipeline_inputs": 8}),
        # config from PipelineLoggingConfig (List[Dict[str, Any])
        (
            PipelineLoggingConfig(
                targets=[TargetLoggingConfig(**obj) for obj in CONFIG_3]
            ),
            14,
            {"pipeline_inputs": 8, "pipeline_outputs": 4},
        ),
        # config from string path to the .yaml file
        (CONFIG_4, 14, {"pipeline_inputs": 8, "pipeline_outputs": 4}),
        # config from PipelineLoggingConfig (yaml str)
        (
            PipelineLoggingConfig(**yaml.safe_load(CONFIG_5)),
            14,
            {"pipeline_inputs": 8, "pipeline_outputs": 4},
        ),
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
        # assert the proper log count for every data log
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
    # assert no superfluous data logs
    assert len([m for m in all_messages if "Category: data" in m]) == sum(
        expected_logs_count.values()
    )
    # assert the proper log count for system logs
    assert (
        len([m for m in all_messages if "Category: system" in m]) == num_iterations * 4
    )
