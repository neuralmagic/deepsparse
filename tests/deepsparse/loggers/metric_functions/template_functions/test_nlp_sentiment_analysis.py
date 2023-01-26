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
from tests.utils import mock_engine
from deepsparse import Pipeline

yaml_config = """
loggers:
    list_logger:
        path: tests/deepsparse/loggers/helpers.py:ListLogger
data_logging:
    predefined:
    - func: sentiment_analysis
      frequency: 1"""

expected_logs = """identifier:text_classification/pipeline_inputs.sequences__string_length, value:8, category:MetricCategories.DATA"""
expected_logs1 = """identifier:text_classification/pipeline_inputs.sequences__string_length, value:[8, 35], category:MetricCategories.DATA"""
expected_logs2 = """identifier:text_classification/pipeline_inputs.sequences__string_length, value:[[8, 35], [26, 34]], category:MetricCategories.DATA"""


@pytest.mark.parametrize(
    "config, inp, num_iterations, expected_logs",
    [
        (yaml_config, "She said", 1, expected_logs),
        (
            yaml_config,
            ["She said", "Ye, can we get married at the mall?"],
            1,
            expected_logs1,
        ),
        (
            yaml_config,
            [
                ["She said", "Ye, can we get married at the mall?"],
                ["Doctors say I'm the illest", "'cause I'm suffering from realness"],
            ],
            1,
            expected_logs2,
        ),
    ],
)
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config, inp, num_iterations, expected_logs):
    pipeline = Pipeline.create("text_classification", logger=config)
    for _ in range(num_iterations):
        pipeline(sequences=inp)

    logs = pipeline.logger.loggers[0].logger.loggers[0].calls
    data_logging_logs = [log for log in logs if "DATA" in log]
    for log, expected_log in zip(data_logging_logs, expected_logs.splitlines()):
        assert log == expected_log
