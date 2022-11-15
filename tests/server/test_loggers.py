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

from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.server import _build_app
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor
from tests.helpers import find_free_port
from deepsparse.engine import Context
from tests.utils import mock_engine
import pytest

"""
END2END Testing Plan
1. Test that the server can be ran without any loggers runs default loggers # i think this is important
2. Test that custom loggers can be specified # covered in build_logger tests
3. Test one specific target # really capture the logs from the server
4. Test multiple targets # really capture the logs from the server
5. Test regex # really capture the logs from the server
9. One function metric with target loggers # really capture the logs from the server
10. Test prometheus client in the server # really capture the logs from the server

"""
from tests.server.test_endpoints import parse, StrSchema
from deepsparse.server.build_logger import build_logger
from deepsparse.server.server import _add_endpoint
from deepsparse import BaseLogger
from deepsparse.server.config import MetricFunctionConfig
from unittest.mock import Mock
import copy

class SinkLogger(BaseLogger):
    def __init__(self):
        self.calls = []
    def log(self, identifier, value, category):
        self.calls.append(f"identifier:{identifier}, value:{value}, category:{category}")

@pytest.mark.parametrize(
    "loggers, data_logger_config, expected_calls",
    [
        # ({"sink_logger": {"path": "tests/server/test_loggers.py:SinkLogger"}}, [], None) # Make sure that we can only have system logs
        ({"sink_logger": {"path": "tests/server/test_loggers.py:SinkLogger"}}, {"re:pipeline_*": [MetricFunctionConfig(func = "identity")]}, None) # Test regex
    ],
)

class TestLoggers:
    @pytest.fixture()
    def setup(self, loggers, data_logger_config, expected_calls):
        stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni"
        server_config = ServerConfig(num_cores=1, num_workers=1, endpoints=[
                EndpointConfig(
                    route="/predict1",
                    task="text-classification",
                    data_logging=data_logger_config,
                    model=stub)], loggers=loggers)
        a = copy.deepcopy(server_config)
        b = build_logger(a)
        context = Context(
            num_cores=server_config.num_cores,
            num_streams=server_config.num_workers,
        )
        executor = ThreadPoolExecutor(max_workers=context.num_streams)

        with mock_engine(rng_seed=0):
            app = _build_app(server_config)
            _add_endpoint(app=app,
                          context = context,
                          executor = executor,
                          server_config = server_config,
                          endpoint_config =
                EndpointConfig(
                    route="/predict",
                    task="text-classification",
                    data_logging=data_logger_config,
                    model=stub),
                          server_logger = b,
                          )
        client = TestClient(app)

        yield app, client, b

    def test_add_model_endpoint(self, setup):
        app, client, server_logger = setup

        for _ in range(5):
            response = client.post("/predict", json={"sequences": "today is great"})
        assert response.status_code == 200
        assert response.json() == int("5678")


