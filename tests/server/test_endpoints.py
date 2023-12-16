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

from typing import List
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel

import pytest
from deepsparse.server.config import EndpointConfig, ServerConfig, SystemLoggingConfig
from deepsparse.server.deepsparse_server import DeepsparseServer
from deepsparse.server.sagemaker import SagemakerServer
from deepsparse.server.server import ProxyPipeline
from fastapi import FastAPI, Request, UploadFile
from fastapi.testclient import TestClient
from tests.utils import mock_engine


class FromFilesSchema(BaseModel):
    def from_files(self, f):
        # do nothing - this method exists just to test files endpoint logic
        ...


class StrSchema(BaseModel):
    value: str


def run_func(value: str):
    return int(value)


class TestStatusEndpoints:
    @pytest.fixture(scope="class")
    def server_config(self):
        server_config = ServerConfig(
            num_cores=1, num_workers=1, endpoints=[], loggers={}
        )
        yield server_config

    @pytest.fixture(scope="class")
    def client(self, server_config):
        server = DeepsparseServer(server_config=server_config)
        yield TestClient(server._build_app())

    def test_config(self, server_config, client):
        response = client.get("/config")
        loaded = ServerConfig(**response.json())
        assert loaded == server_config

    @pytest.mark.parametrize("route", ["/v2/health/ready", "/v2/health/live"])
    def test_pings_exist(self, client, route):
        response = client.get(route)
        assert response.status_code == 200
        assert response.json()["status"] == "OK"

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
        server_config = ServerConfig(
            num_cores=1, num_workers=1, endpoints=[], loggers={}
        )
        yield server_config

    @pytest.fixture(scope="class")
    def server(self, server_config):
        yield DeepsparseServer(server_config=server_config)

    @pytest.fixture(scope="class")
    def sagemaker_server(self, server_config):
        yield SagemakerServer(server_config=server_config)

    @pytest.fixture(scope="class")
    def app(self, server):
        yield server._build_app()

    @pytest.fixture(scope="class")
    def client(self, app):
        yield TestClient(app)

    def test_add_model_endpoint(
        self, server: DeepsparseServer, app: FastAPI, client: TestClient
    ):
        mock_pipeline = AsyncMock(input_schema=str, output_schema=int)

        server._add_inference_endpoints(
            app=app,
            endpoint_config=Mock(route="/predict/parse_int", task="some_task"),
            pipeline=mock_pipeline,
        )

        assert app.routes[-1].path == "/v2/models/predict/parse_int/infer"
        assert app.routes[-1].response_model is int
        assert app.routes[-1].endpoint.func.__annotations__ == {
            "proxy_pipeline": ProxyPipeline,
            "raw_request": Request,
            "system_logging_config": SystemLoggingConfig,
        }
        assert app.routes[-1].methods == {"POST"}

        for v in ["1234", "5678"]:
            response = client.post(
                "/v2/models/predict/parse_int/infer", json=dict(value=v)
            )
            assert response.status_code == 200

    # TODO: udpate test to include check for input_schema, once v2 pipelines
    # have a way to store/fetch compatible from_file pipelines
    def test_add_model_endpoint_with_from_files(self, server, app):
        mock_pipeline = AsyncMock(output_schema=int)

        server._add_inference_endpoints(
            app,
            endpoint_config=Mock(
                route="/predict/parse_int", task="image_classification"
            ),
            pipeline=mock_pipeline,
        )

        assert app.routes[-2].path == "/v2/models/predict/parse_int/infer"
        assert app.routes[-2].endpoint.func.__annotations__ == {
            "proxy_pipeline": ProxyPipeline,
            "raw_request": Request,
            "system_logging_config": SystemLoggingConfig,
        }

        assert app.routes[-1].path == "/v2/models/predict/parse_int/infer/from_files"
        assert app.routes[-1].endpoint.func.__annotations__ == {
            "proxy_pipeline": ProxyPipeline,
            "system_logging_config": SystemLoggingConfig,
            "request": List[UploadFile],
        }
        assert app.routes[-1].response_model is int
        assert app.routes[-1].methods == {"POST"}

    def test_sagemaker_only_adds_one_endpoint(self, sagemaker_server, app):
        num_routes = len(app.routes)
        sagemaker_server._add_inference_endpoints(
            app,
            endpoint_config=Mock(route="predict/parse_int"),
            pipeline=Mock(input_schema=FromFilesSchema, output_schema=int),
        )
        assert len(app.routes) == num_routes + 1
        num_routes = len(app.routes)

        assert app.routes[-1].path == "/invocations/predict/parse_int/infer/from_files"
        assert app.routes[-1].endpoint.func.__annotations__ == {
            "proxy_pipeline": ProxyPipeline,
            "system_logging_config": SystemLoggingConfig,
            "request": List[UploadFile],
        }

        sagemaker_server._add_inference_endpoints(
            app,
            endpoint_config=Mock(route="predict/parse_int"),
            pipeline=Mock(input_schema=StrSchema, output_schema=int),
        )
        assert len(app.routes) == num_routes + 1
        assert app.routes[-1].path == "/invocations/predict/parse_int/infer"
        assert app.routes[-1].endpoint.func.__annotations__ == {
            "proxy_pipeline": ProxyPipeline,
            "system_logging_config": SystemLoggingConfig,
            "raw_request": Request,
        }

    def test_add_endpoint_with_no_route_specified(self, server, app):
        server._add_inference_endpoints(
            app,
            endpoint_config=EndpointConfig(
                route=None,
                name="test_name",
                task="text-classification",
                model="default",
            ),
            pipeline=Mock(input_schema=StrSchema, output_schema=int),
        )

        assert app.routes[-1].path == "/v2/models/test_name/infer"


class TestActualModelEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        stub = "zoo:bert-large-mnli_wikipedia_bookcorpus-pruned80.4block_quantized"
        server_config = ServerConfig(
            num_cores=1,
            num_workers=1,
            endpoints=[
                EndpointConfig(
                    route="/predict/dynamic-batch",
                    task="text-classification",
                    model=stub,
                    batch_size=1,
                ),
                EndpointConfig(
                    route="/predict/static-batch",
                    task="text-classification",
                    model=stub,
                    batch_size=2,
                ),
            ],
            loggers={},  # do not instantiate any loggers
        )
        with mock_engine(rng_seed=0):
            server = DeepsparseServer(server_config=server_config)
            app = server._build_app()
        yield TestClient(app)

    def test_static_batch_errors_on_wrong_batch_size(self, client):
        # this is okay because we can pad batches now
        client.post(
            "/v2/models/predict/static-batch/infer",
            json={"sequences": "today is great"},
        )

    def test_static_batch_good_request(self, client):
        response = client.post(
            "/v2/models/predict/static-batch/infer",
            json={"sequences": ["today is great", "today is terrible"]},
        )
        assert response.status_code == 200
        output = response.json()
        assert len(output["labels"]) == 2
        assert len(output["scores"]) == 2

    @pytest.mark.parametrize(
        "seqs",
        [
            ["today is great"],
            ["today is great", "today is terrible"],
            ["the first sentence", "the second sentence", "the third sentence"],
        ],
    )
    def test_dynamic_batch_any(self, client, seqs):
        response = client.post(
            "/v2/models/predict/dynamic-batch/infer", json={"sequences": seqs}
        )
        assert response.status_code == 200
        output = response.json()
        assert len(output["labels"]) == len(seqs)
        assert len(output["scores"]) == len(seqs)


class TestDynamicEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        server_config = ServerConfig(
            num_cores=1, num_workers=1, endpoints=[], loggers=None
        )
        with mock_engine(rng_seed=0):
            server = DeepsparseServer(server_config=server_config)
            app = server._build_app(server_config)
            yield TestClient(app)


@mock_engine(rng_seed=0)
def test_dynamic_add_and_remove_endpoint(engine_mock):
    server_config = ServerConfig(num_cores=1, num_workers=1, endpoints=[], loggers={})
    server = DeepsparseServer(server_config=server_config)
    app = server._build_app()
    client = TestClient(app)

    # assert /predict doesn't exist
    assert 404 == client.post("/predict", json=dict(sequences="asdf")).status_code

    # add /predict
    response = client.post(
        "/endpoints",
        json=EndpointConfig(
            task="text-classification", model="default", name="test_model"
        ).dict(),
    )

    assert response.status_code == 200

    response = client.post("/v2/models/test_model/infer", json=dict(sequences="asdf"))
    assert response.status_code == 200

    # remove /predict
    response = client.delete(
        "/endpoints",
        json=EndpointConfig(
            route="/v2/models/test_model/infer",
            task="text-classification",
            model="default",
        ).dict(),
    )
    assert response.status_code == 200
    assert 404 == client.post("/predict", json=dict(sequences="asdf")).status_code
