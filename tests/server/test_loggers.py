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

from collections import Counter
from unittest import mock

from deepsparse import PythonLogger
from deepsparse.loggers.config import PipelineSystemLoggingConfig, SystemLoggingGroup
from deepsparse.server.config import (
    EndpointConfig,
    MetricFunctionConfig,
    ServerConfig,
    ServerSystemLoggingConfig,
)
from deepsparse.server.helpers import server_logger_from_config
from deepsparse.server.server import _build_app
from fastapi.testclient import TestClient
from tests.deepsparse.loggers.helpers import fetch_leaf_logger
from tests.helpers import find_free_port
from tests.test_data.server_test_data import SAMPLE_LOGS_DICT
from tests.utils import mock_engine


logger_identifier = "tests/deepsparse/loggers/helpers.py:ListLogger"
stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni"  # noqa E501
task = "text-classification"
name = "endpoint_name"


def _test_logger_contents(leaf_logger, expected_logs):
    for expected_log_content in list(expected_logs.keys()):
        i = 0
        for log in leaf_logger.calls:
            if expected_log_content in log:
                i += 1
        assert expected_logs[expected_log_content] == i


def test_default_logger():
    server_config = ServerConfig(
        endpoints=[EndpointConfig(task=task, name=name, model=stub)]
    )
    server_logger = server_logger_from_config(server_config)
    with mock.patch(
        "deepsparse.server.server.server_logger_from_config", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)

    for _ in range(2):
        client.post("/predict", json={"sequences": "today is great"})
    assert isinstance(fetch_leaf_logger(server_logger), PythonLogger)


def test_logging_only_system_info():
    server_config = ServerConfig(
        endpoints=[EndpointConfig(task=task, name=name, model=stub)],
        loggers={"logger_1": {"path": logger_identifier}},
    )
    server_logger = server_logger_from_config(server_config)
    with mock.patch(
        "deepsparse.server.server.server_logger_from_config", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)

    for _ in range(2):
        client.post("/predict", json={"sequences": "today is great"})
    _test_logger_contents(
        fetch_leaf_logger(server_logger),
        {"prediction_latency": 8},
    )


def test_regex_target_logging():
    server_config = ServerConfig(
        endpoints=[
            EndpointConfig(
                task=task,
                name=name,
                data_logging={
                    "re:.*pipeline*.": [MetricFunctionConfig(func="identity")]
                },
                model=stub,
            )
        ],
        loggers={"logger_1": {"path": logger_identifier}},
    )
    server_logger = server_logger_from_config(server_config)
    with mock.patch(
        "deepsparse.server.server.server_logger_from_config", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)

    for _ in range(2):
        client.post("/predict", json={"sequences": "today is great"})
    _test_logger_contents(
        fetch_leaf_logger(server_logger),
        {"pipeline_inputs__identity": 2, "pipeline_outputs__identity": 2},
    )


def test_multiple_targets_logging():
    server_config = ServerConfig(
        endpoints=[
            EndpointConfig(
                task=task,
                name=name,
                data_logging={
                    "pipeline_inputs.sequences": [
                        MetricFunctionConfig(func="identity")
                    ],
                    "engine_inputs": [MetricFunctionConfig(func="identity")],
                },
                model=stub,
            )
        ],
        loggers={"logger_1": {"path": logger_identifier}},
    )
    server_logger = server_logger_from_config(server_config)
    with mock.patch(
        "deepsparse.server.server.server_logger_from_config", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)

    for _ in range(2):
        client.post("/predict", json={"sequences": "today is great"})
    _test_logger_contents(
        fetch_leaf_logger(server_logger),
        {
            "pipeline_inputs.sequences__identity": 2,
            "engine_inputs__identity": 2,
            "prediction_latency": 8,
        },
    )


def test_function_metric_with_target_loggers():
    server_config = ServerConfig(
        endpoints=[
            EndpointConfig(
                task=task,
                name=name,
                data_logging={
                    "pipeline_inputs.sequences[0]": [
                        MetricFunctionConfig(
                            func="identity", target_loggers=["logger_1"]
                        )
                    ],
                    "engine_inputs": [MetricFunctionConfig(func="identity")],
                },
                model=stub,
            )
        ],
        loggers={
            "logger_1": {"path": logger_identifier},
            "logger_2": {"path": logger_identifier},
        },
    )
    server_logger = server_logger_from_config(server_config)
    with mock.patch(
        "deepsparse.server.server.server_logger_from_config", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)

    for _ in range(2):
        client.post("/predict", json={"sequences": "today is great"})

    _test_logger_contents(
        server_logger.logger.loggers[1].logger.loggers[0],
        {
            "pipeline_inputs.sequences__identity": 2,
            "engine_inputs__identity": 2,
            "prediction_latency": 8,
        },
    )
    _test_logger_contents(
        server_logger.logger.loggers[1].logger.loggers[1],
        {
            "pipeline_inputs.sequences__identity": 0,
            "engine_inputs__identity": 2,
            "prediction_latency": 8,
        },
    )


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


@mock_engine(rng_seed=0)
def test_endpoint_system_logging(tmp_path):
    server_config = ServerConfig(
        system_logging=ServerSystemLoggingConfig(
            request_details=SystemLoggingGroup(enable=True),
            resource_utilization=SystemLoggingGroup(enable=True),
        ),
        endpoints=[
            EndpointConfig(
                task="text_classification",
                model="default",
                route="/predict_text_classification",
                logging_config=PipelineSystemLoggingConfig(
                    inference_details=SystemLoggingGroup(enable=True),
                    prediction_latency=SystemLoggingGroup(enable=True),
                ),
            ),
            EndpointConfig(
                task="question_answering",
                model="default",
                route="/predict_question_answering",
                logging_config=PipelineSystemLoggingConfig(
                    inference_details=SystemLoggingGroup(enable=True),
                    prediction_latency=SystemLoggingGroup(enable=True),
                ),
            ),
        ],
        loggers={"logger_1": {"path": logger_identifier}},
    )
    server_logger = server_logger_from_config(server_config)
    with mock.patch(
        "deepsparse.server.server.server_logger_from_config", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)
    client.post("/predict_text_classification", json=dict(sequences="asdf"))
    client.post(
        "/predict_text_classification", json=dict(question="asdf", context="asdf")
    )
    calls = server_logger.logger.loggers[0].logger.loggers[0].calls

    c = Counter([call.split(",")[0] for call in calls])

    assert c == SAMPLE_LOGS_DICT
