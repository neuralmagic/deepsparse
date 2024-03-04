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

from typing import Dict, List

import pytest
from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.deepsparse_server import DeepsparseServer
from fastapi.testclient import TestClient


TEST_MODEL_ID = "hf:mgoin/TinyStories-1M-ds"


@pytest.fixture(scope="module")
def endpoint_config():
    endpoint = EndpointConfig(
        task="text_generation", model=TEST_MODEL_ID, middlewares=["TimerMiddleware"]
    )
    return endpoint


@pytest.fixture(scope="module")
def server_config(endpoint_config):
    server_config = ServerConfig(
        num_cores=1, num_workers=1, endpoints=[endpoint_config], loggers={}
    )

    return server_config


@pytest.fixture(scope="module")
def server(server_config):
    server = DeepsparseServer(server_config=server_config)
    return server


@pytest.fixture(scope="module")
def app(server):
    app = server._build_app()
    return app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


def test_benchmark_pipeline(client):
    url = "v2/models/text_generation-0/benchmark"
    obj = {
        "data_type": "dummy",
        "gen_sequence_length": 100,
        "pipeline_kwargs": {},
        "input_schema_kwargs": {},
    }
    response = client.post(url, json=obj)
    response.raise_for_status()

    response_json: List[Dict] = response.json()

    # iterate over all benchmarks that are Lists
    # The final layer is a dict, where
    # key is the name of the Operator, value is the List of timings
    # Ex. {'CompileGeneratedTokens': [1.7404556274414062e-05, ... }
    timings = response_json[0]
    for timing in timings:
        for key, values in timing.items():
            assert isinstance(key, str)
            assert isinstance(values, List)
            for run_time in values:
                assert isinstance(run_time, float)
