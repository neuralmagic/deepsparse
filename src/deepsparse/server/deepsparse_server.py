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
from deepsparse.server.config import EndpointConfig
from deepsparse.server.server import CheckReady, ModelMetaData, ProxyPipeline, Server
from fastapi import FastAPI


_LOGGER = logging.getLogger(__name__)


class DeepsparseServer(Server):
    def _add_routes(self, app):
        @app.get("/v2/health/ready", tags=["health"], response_model=CheckReady)
        @app.get("/v2/health/live", tags=["health"], response_model=CheckReady)
        def _check_health():
            return CheckReady(status="OK")

        @app.get("/v2", tags=["metadata", "server"], response_model=str)
        def _get_server_info():
            return "This is the deepsparse server. Hello!"

        @app.post("/endpoints", tags=["endpoints"], response_model=bool)
        def _add_endpoint_endpoint(cfg: EndpointConfig):
            if cfg.name is None:
                cfg.name = f"endpoint-{len(app.routes)}"
            self._add_endpoint(
                app,
                cfg,
            )
            # force regeneration of the docs
            app.openapi_schema = None
            return True

        @app.delete("/endpoints", tags=["endpoints"], response_model=bool)
        def _delete_endpoint(cfg: EndpointConfig):
            _LOGGER.info(f"Deleting endpoint for {cfg}")
            matching = [r for r in app.routes if r.path == cfg.route]
            assert len(matching) == 1
            app.routes.remove(matching[0])
            # force regeneration of the docs
            app.openapi_schema = None
            return True

        for endpoint_config in self.server_config.endpoints:
            self._add_endpoint(
                app,
                endpoint_config,
            )

        _LOGGER.info(f"Added endpoints: {[route.path for route in app.routes]}")
        return app

    def _add_endpoint(
        self,
        app: FastAPI,
        endpoint_config: EndpointConfig,
    ):
        pipeline_config = endpoint_config.to_pipeline_config()
        pipeline_config.kwargs["executor"] = self.executor

        _LOGGER.info(f"Initializing pipeline for '{endpoint_config.name}'")
        pipeline = Pipeline.from_config(
            pipeline_config, self.context, self.server_logger
        )

        _LOGGER.info(f"Adding endpoints for '{endpoint_config.name}'")
        self._add_inference_endpoints(
            app,
            endpoint_config,
            pipeline,
        )
        self._add_status_and_metadata_endpoints(app, endpoint_config, pipeline)

    def _add_status_and_metadata_endpoints(
        self,
        app: FastAPI,
        endpoint_config: EndpointConfig,
        pipeline: Pipeline,
    ):
        routes_and_fns = []
        meta_and_fns = []

        if endpoint_config.route:
            endpoint_config.route = self.clean_up_route(endpoint_config.route)
            route_ready = f"/v2/models{endpoint_config.route}/ready"
            route_meta = f"/v2/models{endpoint_config.route}"
        else:
            route_ready = f"/v2/models/{endpoint_config.name}/ready"
            route_meta = f"/v2/models/{endpoint_config.name}"

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

    def _add_inference_endpoints(
        self,
        app: FastAPI,
        endpoint_config: EndpointConfig,
        pipeline: Pipeline,
    ):
        routes_and_fns = []
        if endpoint_config.route:
            endpoint_config.route = self.clean_up_route(endpoint_config.route)
            route = f"/v2/models{endpoint_config.route}/infer"
        else:
            route = f"/v2/models/{endpoint_config.name}/infer"

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

        self._update_routes(
            app=app,
            routes_and_fns=routes_and_fns,
            response_model=pipeline.output_schema,
            methods=["POST"],
            tags=["model", "inference"],
        )
