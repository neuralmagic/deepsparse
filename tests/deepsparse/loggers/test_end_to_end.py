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

import yaml

import pytest
from deepsparse import Pipeline, logger_from_config
from deepsparse.loggers.config import PipelineLoggingConfig
from tests.utils import mock_engine


YAML_CONFIG = """
    loggers:
        list_logger:
            path: tests/deepsparse/loggers/helpers.py:ListLogger
    system_logging:
        enable: true
        prediction_latency:
            enable: true
        resource_utilization:
            enable: true
    data_logging:
        pipeline_inputs.sequences[0]:
            - func: identity"""

PIPELINE_LOGGING_CONFIG = PipelineLoggingConfig(**yaml.safe_load(YAML_CONFIG))
PATH = "tests/test_data/logging_config.yaml"


@pytest.mark.parametrize("config", [YAML_CONFIG, PIPELINE_LOGGING_CONFIG, PATH])
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config):
    logger = logger_from_config(config)

    pipeline = Pipeline.create("text_classification", logger=logger)

    for i in range(10):
        pipeline("today is great")

    time.sleep(1)  # sleeping to make sure all the logs are collected
    calls = pipeline.logger.logger.loggers[0].logger.loggers[0].calls
    data_logging_calls = [call for call in calls if "DATA" in call]
    assert len(data_logging_calls) == 10
    prediction_latency_calls = [call for call in calls if "SYSTEM" in call]
    assert len(prediction_latency_calls) == 41
