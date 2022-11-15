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

from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.server import _build_app
from fastapi.testclient import TestClient
from tests.helpers import find_free_port
from tests.utils import mock_engine
import pytest

"""
END2END Testing Plan
1. Test that the server can be ran without any loggers runs default loggers # i think this is important
2. Test that custom loggers can be specified # covered in build_logger tests
3. Test one specific target # really capture the logs from the server
4. Test multiple targets # really capture the logs from the server
5. Test regex # really capture the logs from the server
7. Make sure that we can only have system logs # really capture the logs from the server
9. One function metric with target loggers # really capture the logs from the server
10. Test prometheus client in the server # really capture the logs from the server

"""


class TestMockEndpoints:
    @pytest.fixture(scope="class")
    def server_config(self):
        server_config = ServerConfig(
            num_cores=1, num_workers=1, endpoints=[], loggers={}
        )
        yield server_config

    @pytest.fixture(scope="class")
    def app(self, server_config):
        yield _build_app(server_config)

    @pytest.fixture(scope="class")
    def client(self, app):
        yield TestClient(app)

    def test_add_model_endpoint(self, app: FastAPI, client: TestClient):
        mock_pipeline = Mock(
            side_effect=parse,
            input_schema=StrSchema,
            output_schema=int,
            logger=MultiLogger([]),
        )
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(route="/predict/parse_int"),
            pipeline=mock_pipeline,
        )
        assert app.routes[-1].path == "/predict/parse_int"
        assert app.routes[-1].response_model is int
        assert app.routes[-1].endpoint.__annotations__ == {"request": StrSchema}
        assert app.routes[-1].methods == {"POST"}

        for v in ["1234", "5678"]:
            response = client.post("/predict/parse_int", json=dict(value=v))
            assert response.status_code == 200
            assert response.json() == int(v)

    def test_add_model_endpoint_with_from_files(self, app):
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(route="/predict/parse_int"),
            pipeline=Mock(input_schema=FromFilesSchema, output_schema=int),
        )
        assert app.routes[-2].path == "/predict/parse_int"
        assert app.routes[-2].endpoint.__annotations__ == {"request": FromFilesSchema}
        assert app.routes[-1].path == "/predict/parse_int/from_files"
        assert app.routes[-1].endpoint.__annotations__ == {"request": List[UploadFile]}
        assert app.routes[-1].response_model is int
        assert app.routes[-1].methods == {"POST"}

    def test_sagemaker_only_adds_one_endpoint(self, app):
        num_routes = len(app.routes)
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(route="/predict/parse_int"),
            pipeline=Mock(input_schema=FromFilesSchema, output_schema=int),
            integration="sagemaker",
        )
        assert len(app.routes) == num_routes + 1
        assert app.routes[-1].path == "/invocations"
        assert app.routes[-1].endpoint.__annotations__ == {"request": List[UploadFile]}

        num_routes = len(app.routes)
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(route="/predict/parse_int"),
            pipeline=Mock(input_schema=StrSchema, output_schema=int),
            integration="sagemaker",
        )
        assert len(app.routes) == num_routes + 1
        assert app.routes[-1].path == "/invocations"
        assert app.routes[-1].endpoint.__annotations__ == {"request": StrSchema}

    def test_add_endpoint_with_no_route_specified(self, app):
        _add_pipeline_endpoint(
            app,
            endpoint_config=Mock(route=None),
            pipeline=Mock(input_schema=StrSchema, output_schema=int),
        )
        assert app.routes[-1].path == "/predict"
