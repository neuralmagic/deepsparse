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

import pytest
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import (
    EndpointConfig,
    ServerConfig,
    SystemLoggingConfig,
    SystemLoggingGroup,
)
from deepsparse.server.server import _build_app
from fastapi.testclient import TestClient
from tests.utils import mock_engine


logger_identifier = "tests/deepsparse/loggers/helpers.py:ListLogger"
stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni"  # noqa E501
task = "text-classification"
name = "endpoint_name"


@pytest.mark.parametrize(
    "json_payload, batch_size, successful_request",
    [
        ({"sequences": "today is great"}, 1, True),
        ({"sequences": ["today is great", "today is great"]}, 2, True),
        ({"this": "is supposed to fail"}, 1, False),
    ],
)
def test_log_request_details(json_payload, batch_size, successful_request):
    server_config = ServerConfig(
        endpoints=[
            EndpointConfig(task=task, name=name, model=stub, batch_size=batch_size)
        ],
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
    client.post("/predict", json=json_payload)

    calls = server_logger.logger.loggers[0].logger.loggers[0].calls

    successful_request_calls = [call for call in calls if "successful_request" in call]
    assert len(successful_request_calls) == 1
    successful_request_logged = int(
        successful_request_calls[0].split("value:")[1].split(",")[0]
    )
    assert bool(successful_request_logged) == successful_request

    if successful_request:
        batch_size_calls = [call for call in calls if "batch_size" in call]
        assert len(batch_size_calls) == 1
        batch_size_logged = int(batch_size_calls[0].split("value:")[1].split(",")[0])
        assert batch_size_logged == server_config.endpoints[0].batch_size
