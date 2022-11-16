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

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from deepsparse import PythonLogger
from deepsparse.engine import Context
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import EndpointConfig, MetricFunctionConfig, ServerConfig
from deepsparse.server.server import _add_endpoint, _build_app
from fastapi.testclient import TestClient
from tests.helpers import find_free_port
from tests.utils import mock_engine


LIST_LOGGER_IDENTIFIER = "tests/deepsparse/loggers/helpers.py:ListLogger"

EXPECTED_LOGS_SYSTEM = {"category:MetricCategories.SYSTEM": 8}
EXPECTED_LOGS_REGEX = {"pipeline_inputs.identity": 2, "pipeline_outputs.identity": 2}
EXPECTED_LOGS_MULTIPLE_TARGETS = {
    "pipeline_inputs.sequences.identity": 2,
    "engine_inputs.identity": 2,
}
EXPECTED_LOGS_TARGET_LOGGER = [
    {**EXPECTED_LOGS_SYSTEM, **EXPECTED_LOGS_MULTIPLE_TARGETS},
    {
        **EXPECTED_LOGS_SYSTEM,
        **{"pipeline_inputs.sequences.identity": 0, "engine_inputs.identity": 2},
    },
]

DATA_LOGGER_CONFIG_REGEX = {"re:.*pipeline*.": [MetricFunctionConfig(func="identity")]}
DATA_LOGGER_CONFIG_MULTIPLE_TARGETS = {
    "pipeline_inputs.sequences": [MetricFunctionConfig(func="identity")],
    "engine_inputs": [MetricFunctionConfig(func="identity")],
}
DATA_LOGGER_CONFIG_TARGET_LOGGER = {
    "pipeline_inputs.sequences": [
        MetricFunctionConfig(func="identity", target_loggers=["logger_1"])
    ],
    "engine_inputs": [MetricFunctionConfig(func="identity")],
}


def _test_logger_contents(leaf_logger, expected_logs):
    for expected_log_content in list(expected_logs.keys()):
        i = 0
        for log in leaf_logger.calls:
            if expected_log_content in log:
                i += 1
        assert expected_logs[expected_log_content] == i


@pytest.mark.parametrize(
    "loggers, data_logger_config, expected_logs_content",
    [
        ({}, [], None),  # Test default logger (no `loggers` specified)
        (
            {"logger_1": {"path": LIST_LOGGER_IDENTIFIER}},
            [],
            EXPECTED_LOGS_SYSTEM,
        ),  # Make sure that we log system logs only
        (
            {"logger_1": {"path": LIST_LOGGER_IDENTIFIER}},
            DATA_LOGGER_CONFIG_REGEX,
            {**EXPECTED_LOGS_SYSTEM, **EXPECTED_LOGS_REGEX},
        ),  # Make sure we can use regex to target specific identifiers
        (
            {"logger_1": {"path": LIST_LOGGER_IDENTIFIER}},
            DATA_LOGGER_CONFIG_MULTIPLE_TARGETS,
            EXPECTED_LOGS_MULTIPLE_TARGETS,
        ),  # Test multiple targets
        (
            {
                "logger_1": {"path": LIST_LOGGER_IDENTIFIER},
                "logger_2": {"path": LIST_LOGGER_IDENTIFIER},
            },
            DATA_LOGGER_CONFIG_TARGET_LOGGER,
            EXPECTED_LOGS_TARGET_LOGGER,
        ),  # One function metric with target loggers
    ],
)
class TestLoggers:
    @pytest.fixture()
    def setup(self, loggers, data_logger_config, expected_logs_content):

        stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni"  # noqa E501
        task = "text-classification"
        name = "endpoint_name"

        server_config = ServerConfig(
            endpoints=[
                EndpointConfig(
                    task=task,
                    data_logging=data_logger_config,
                    name=name,
                    model=stub,
                    route="/predict_",
                )
            ],
            loggers=loggers,
        )

        # create a duplicate of `server_logger` to later add it together
        # with a separate endpoint. The goal is to have access to the
        # `server_logger` and inspect its state change during server inference
        server_logger = build_logger(copy.deepcopy(server_config))

        with mock.patch(
            "deepsparse.server.server.build_logger", return_value=server_logger
        ), mock_engine(rng_seed=0):
            app = _build_app(server_config)

        client = TestClient(app)

        yield client, server_logger, expected_logs_content

    def test_logger_contents(self, setup):
        client, server_logger, expected_logs_content = setup
        for _ in range(2):
            client.post("/predict_", json={"sequences": "today is great"})

        if expected_logs_content is None:
            assert isinstance(server_logger.logger.loggers, PythonLogger)

        elif isinstance(expected_logs_content, list):
            leaf_loggers = server_logger.logger.loggers[1].logger.loggers
            for idx, leaf_logger in enumerate(leaf_loggers):
                _test_logger_contents(leaf_logger, expected_logs_content[idx])
        else:
            leaf_logger = server_logger.logger.loggers[0].logger.loggers[
                0
            ]  # -> MultiLogger -> FunctionLogger -> MultiLogger
            _test_logger_contents(leaf_logger, expected_logs_content)


@mock_engine(rng_seed=0)
def test_instantiate_prometheus(tmp_path):
    client = TestClient(
        _build_app(
            ServerConfig(
                endpoints=[EndpointConfig(task="text_classification", model="default")],
                loggers=dict(
                    prometheus={
                        "port": find_free_port(),
                        "text_log_save_dir": str(tmp_path),
                        "text_log_save_frequency": 30,
                    }
                ),
            )
        )
    )
    r = client.post("/predict", json=dict(sequences="asdf"))
    assert r.status_code == 200
