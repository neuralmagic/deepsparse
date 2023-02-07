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

from unittest import mock

from deepsparse import PythonLogger
from deepsparse.server.config import EndpointConfig, MetricFunctionConfig, ServerConfig
from deepsparse.server.helpers import server_logger_from_config
from deepsparse.server.server import _build_app
from fastapi.testclient import TestClient
from tests.deepsparse.loggers.helpers import fetch_leaf_logger
from tests.helpers import find_free_port
from tests.utils import mock_engine


logger_identifier = "tests/deepsparse/loggers/helpers.py:ListLogger"
stub = "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/12layer_pruned80_quant-none-vnni"  # noqa E501
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


def test_data_logging_from_predefined():
    server_config = ServerConfig(
        endpoints=[
            EndpointConfig(
                task=task,
                name="text_classification",
                model=stub,
                add_predefined=[MetricFunctionConfig(func="text_classification")],
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
    client.post(
        "/predict",
        json={
            "sequences": [["Fun for adults and children.", "Fun for only children."]]
        },
    )
    calls = fetch_leaf_logger(server_logger).calls
    data_logging_logs = [call for call in calls if "DATA" in call]
    with open(
        "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/text_classification.txt",  # noqa E501
        "r",
    ) as f:
        expected_logs = f.read().splitlines()
    for log, expected_log in zip(data_logging_logs, expected_logs):
        assert log == expected_log


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
