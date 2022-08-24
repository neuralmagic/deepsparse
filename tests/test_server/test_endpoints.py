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

from unittest.mock import Mock

from pydantic import BaseModel

import pytest
from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.server import _add_pipeline_endpoint, _build_app
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FromFilesSchema(BaseModel):
    def from_files(self, f):
        # do nothing - this method exists just to test files endpoint logic
        ...


class StrSchema(BaseModel):
    value: str


def parse(v: StrSchema) -> int:
    return int(v.value)


class TestStatusEndpoints:
    @pytest.fixture(scope="class")
    def server_config(self):
        server_config = ServerConfig(num_cores=1, num_workers=1, endpoints=[])
        yield server_config

    @pytest.fixture(scope="class")
    def client(self, server_config):
        yield TestClient(_build_app(server_config))

    def test_config(self, server_config, client):
        response = client.get("/config")
        loaded = ServerConfig(**response.json())
        assert loaded == server_config

    @pytest.mark.parametrize(
        "endpoint", ["/ping", "/health", "/healthcheck", "/status"]
    )
    def test_pings_exist(self, client, endpoint):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json() is True

    def test_docs_exist(self, client):
        assert client.get("/docs").status_code == 200

    def test_home_redirects_to_docs(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.request.path_url == "/docs"
        assert len(response.history) > 0
        assert response.history[-1].is_redirect


class TestMockEndpoints:
    @pytest.fixture(scope="class")
    def server_config(self):
        server_config = ServerConfig(num_cores=1, num_workers=1, endpoints=[])
        yield server_config

    @pytest.fixture(scope="class")
    def app(self, server_config):
        yield _build_app(server_config)

    @pytest.fixture(scope="class")
    def client(self, app):
        yield TestClient(app)

    def test_add_model_endpoint(self, app: FastAPI, client: TestClient):
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(endpoint="/predict/parse_int"),
            pipeline=Mock(input_schema=StrSchema, output_schema=int, side_effect=parse),
        )
        assert app.routes[-1].path == "/predict/parse_int"
        assert app.routes[-1].response_model is int
        assert app.routes[-1].methods == {"POST"}

        for v in ["1234", "5678"]:
            response = client.post("/predict/parse_int", json=dict(value=v))
            assert response.status_code == 200
            assert response.json() == int(v)

    def test_add_model_endpoint_with_from_files(self, app):
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(endpoint="/predict/parse_int"),
            pipeline=Mock(input_schema=FromFilesSchema, output_schema=int),
        )
        assert app.routes[-1].path == "/predict/parse_int/files"
        assert app.routes[-1].response_model is int
        assert app.routes[-1].methods == {"POST"}

    def test_add_invocations_endpoint(self, app):
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(endpoint="/predict/parse_int"),
            pipeline=Mock(input_schema=FromFilesSchema, output_schema=int),
            add_invocations_endpoint=True,
        )
        assert app.routes[-1].path == "/invocations"

    def test_add_endpoint_with_no_route_specified(self, app):
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(endpoint=None),
            pipeline=Mock(input_schema=StrSchema, output_schema=int),
        )
        assert app.routes[-1].path == "/predict"


class TestActualModelEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        stub = (
            "zoo:nlp/text_classification/distilbert-none/"
            "pytorch/huggingface/qqp/pruned80_quant-none-vnni"
        )
        server_config = ServerConfig(
            num_cores=1,
            num_workers=1,
            endpoints=[
                EndpointConfig(
                    endpoint="/predict/dynamic-batch",
                    task="text-classification",
                    model=stub,
                    batch_size=1,
                ),
                EndpointConfig(
                    endpoint="/predict/static-batch",
                    task="text-classification",
                    model=stub,
                    batch_size=2,
                ),
            ],
        )
        app = _build_app(server_config)
        yield TestClient(app)

    def test_static_batch_errors_on_wrong_batch_size(self, client):
        with pytest.raises(
            ValueError, match="batch size of 1 must match the batch size"
        ):
            client.post("/predict/static-batch", json={"sequences": "today is great"})

    def test_static_batch_good_request(self, client):
        response = client.post(
            "/predict/static-batch",
            json={"sequences": ["today is great", "today is terrible"]},
        )
        assert response.status_code == 200
        output = response.json()
        assert len(output["labels"]) == 2
        assert len(output["scores"]) == 2
