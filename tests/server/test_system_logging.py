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

from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import (
    EndpointConfig,
    ServerConfig,
    SystemLoggingConfig,
    SystemLoggingGroup,
)
from deepsparse.server.server import _build_app
from fastapi.testclient import TestClient
from tests.deepsparse.loggers.helpers import ListLogger
from tests.utils import mock_engine


logger_identifier = "tests/deepsparse/loggers/helpers.py:ListLogger"
stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni"  # noqa E501
task = "text-classification"
name = "endpoint_name"


def test_end_to_end():
    server_config = ServerConfig(
        endpoints=[EndpointConfig(task=task, name=name, model=stub)],
        loggers={"logger_1": {"path": logger_identifier}},
        system_logging=SystemLoggingConfig(
            request_details=SystemLoggingGroup(enable=True)
        ),
    )
    server_logger = build_logger(server_config)
    with mock.patch(
        "deepsparse.server.server.build_logger", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)
    client.post("/predict", json={"sequences_": "today is great"})

    calls = server_logger.logger.loggers[0].logger.loggers[0].calls
    assert (
        len(
            [call for call in calls if call.startswith("identifier:prediction_latency")]
        )
        == 4
    )
    assert (
        len([call for call in calls if call.startswith("identifier:request_details")])
        == 1
    )
