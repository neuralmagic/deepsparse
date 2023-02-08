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

import time

import pytest
from deepsparse import Pipeline, logger_from_config
from tests.utils import mock_engine


YAML_CONFIG = """
    loggers:
        list_logger:
            path: tests/deepsparse/loggers/helpers.py:ListLogger
    system_logging:
        enable: true
        prediction_latency:
            enable: true
        inference_details:
            enable: true
    data_logging:
        {possible_pipeline_name}pipeline_inputs.sequences[0]:
            - func: identity"""


LOGGER = logger_from_config(
    YAML_CONFIG.format(possible_pipeline_name="text_classification/")
)


@pytest.mark.parametrize(
    "config",
    [
        YAML_CONFIG.format(possible_pipeline_name=""),
        YAML_CONFIG.format(possible_pipeline_name="text_classification/"),
        LOGGER,
    ],
)
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config):
    pipeline = Pipeline.create("text_classification", logger=config)
    no_iterations = 10
    for i in range(no_iterations):
        pipeline("today is great")
        time.sleep(0.1)  # sleeping to make sure all the logs are collected

    calls = pipeline.logger.logger.loggers[0].logger.loggers[0].calls
    data_logging_calls = [call for call in calls if "DATA" in call]
    assert len(data_logging_calls) == no_iterations
    prediction_latency_calls = [call for call in calls if "prediction_latency" in call]
    assert len(prediction_latency_calls) == 4 * no_iterations
    inference_details_calls = [call for call in calls if "inference_details" in call]
    assert len(inference_details_calls) == 1 + no_iterations
