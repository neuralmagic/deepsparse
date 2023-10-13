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

import logging
from functools import partial

from deepsparse import Pipeline
from deepsparse.server.deepsparse_server import DeepsparseServer, EndpointConfig
from deepsparse.server.server import CheckReady, ModelMetaData, ProxyPipeline, Server
from fastapi import FastAPI


_LOGGER = logging.getLogger(__name__)


class SagemakerServer(DeepsparseServer):
    """
    Server support for sagemaker. This integration support includes all the routes
    supported by the Deepsparse server, with the additional route /ping. All sagemaker
    endpoints also start with /invocations. All endpoints are added/deleted in the
    same way as the Deepsparse server.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_routes(self, app: FastAPI):
        @app.get("/ping", tags=["health"], response_model=CheckReady)
        def _check_health():
            return CheckReady(status="OK")

        return super()._add_routes(app)

    def _add_inference_endpoints(
        self,
        app: FastAPI,
        endpoint_config: EndpointConfig,
        pipeline: Pipeline,
    ):
        routes_and_fns = []
        if endpoint_config.route:
            endpoint_config.route = self.clean_up_route(endpoint_config.route)
            route = f"/invocations{endpoint_config.route}/infer"
        else:
            route = f"/invocations/{endpoint_config.name}/infer"

        if hasattr(pipeline.input_schema, "from_files"):
            routes_and_fns.append(
                (
                    route + "/from_files",
                    partial(
                        Server.predict_from_files,
                        ProxyPipeline(pipeline),
                        self.server_config.system_logging,
                    ),
                )
            )
        else:
            routes_and_fns.append(
                (
                    route,
                    partial(
                        Server.predict,
                        ProxyPipeline(pipeline),
                        self.server_config.system_logging,
                    ),
                )
            )

        self._update_routes(
            app=app,
            routes_and_fns=routes_and_fns,
            response_model=pipeline.output_schema,
            methods=["POST"],
            tags=["model", "inference"],
        )

    def _add_status_and_metadata_endpoints(
        self, app: FastAPI, endpoint_config: EndpointConfig, pipeline: Pipeline
    ):
        routes_and_fns = []
        meta_and_fns = []

        if endpoint_config.route:
            endpoint_config.route = self.clean_up_route(endpoint_config.route)
            route_ready = f"/invocations{endpoint_config.route}/ready"
            route_meta = f"/invocations{endpoint_config.route}"
        else:
            route_ready = f"/invocations/{endpoint_config.name}/ready"
            route_meta = f"/invocations/{endpoint_config.name}"

        routes_and_fns.append((route_ready, Server.pipeline_ready))
        meta_and_fns.append(
            (route_meta, partial(Server.model_metadata, ProxyPipeline(pipeline)))
        )

        self._update_routes(
            app=app,
            routes_and_fns=meta_and_fns,
            response_model=ModelMetaData,
            methods=["GET"],
            tags=["model", "metadata"],
        )
        self._update_routes(
            app=app,
            routes_and_fns=routes_and_fns,
            response_model=CheckReady,
            methods=["GET"],
            tags=["model", "health"],
        )
