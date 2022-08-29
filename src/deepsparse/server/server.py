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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List

import yaml

import uvicorn
from deepsparse.engine import Context
from deepsparse.pipeline import Pipeline
from deepsparse.server.config import (
    INTEGRATION_LOCAL,
    INTEGRATION_SAGEMAKER,
    INTEGRATIONS,
    EndpointConfig,
    ServerConfig,
)
from fastapi import FastAPI, UploadFile
from starlette.responses import RedirectResponse


_LOGGER = logging.getLogger(__name__)


def start_server(
    config_path: str, host: str = "0.0.0.0", port: int = 5543, log_level: str = "info"
):
    """
    Starts a FastAPI server with uvicorn with the configuration specified.

    :param config_path: A yaml file with the server config. See :class:`ServerConfig`.
    :param host: The IP address to bind the server to.
    :param port: The port to listen on.
    :param log_level: Log level given to python and uvicorn logging modules.
    """
    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"][__name__] = {
        "handlers": ["default"],
        "level": log_level.upper(),
    }

    with open(config_path) as fp:
        obj = yaml.safe_load(fp)
    server_config = ServerConfig(**obj)
    _LOGGER.info(f"Using config: {repr(server_config)}")

    app = _build_app(server_config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        log_config=log_config,
        # NOTE: only want to have 1 server process so models aren't being copied
        workers=1,
    )


def _build_app(server_config: ServerConfig) -> FastAPI:
    route_counts = Counter([cfg.route for cfg in server_config.endpoints])
    if route_counts[None] > 1:
        raise ValueError(
            "You must specify `route` for all endpoints if multiple endpoints are used."
        )

    for route, count in route_counts.items():
        if count > 1:
            raise ValueError(
                f"{route} specified {count} times for multiple EndpoingConfig.route"
            )

    if server_config.integration not in INTEGRATIONS:
        raise ValueError(
            f"Unknown integration field {server_config.integration}. "
            f"Expected one of {INTEGRATIONS}"
        )

    context = Context(
        num_cores=server_config.num_cores,
        num_streams=server_config.num_workers,
    )
    executor = ThreadPoolExecutor(max_workers=context.num_streams)

    _LOGGER.info(f"Built context: {repr(context)}")
    _LOGGER.info(f"Built ThreadPoolExecutor with {executor._max_workers} workers")

    app = FastAPI()

    @app.get("/", include_in_schema=False)
    def _home():
        return RedirectResponse("/docs")

    @app.get("/config", tags=["general"], response_model=ServerConfig)
    def _info():
        return server_config

    @app.get("/ping", tags=["general"], response_model=bool)
    @app.get("/health", tags=["general"], response_model=bool)
    @app.get("/healthcheck", tags=["general"], response_model=bool)
    @app.get("/status", tags=["general"], response_model=bool)
    def _health():
        return True

    # fill in names if there are none
    for idx, endpoint_config in enumerate(server_config.endpoints):
        if endpoint_config.name is None:
            endpoint_config.name = f"endpoint-{idx}"

    # create pipelines & endpoints
    for endpoint_config in server_config.endpoints:
        pipeline_config = endpoint_config.to_pipeline_config()
        pipeline_config.kwargs["executor"] = executor

        _LOGGER.info(f"Initializing pipeline for '{endpoint_config.name}'")
        pipeline = Pipeline.from_config(pipeline_config, context)

        _LOGGER.info(f"Adding endpoints for '{endpoint_config.name}'")
        _add_pipeline_endpoint(
            app, endpoint_config, pipeline, server_config.integration
        )

    _LOGGER.info(f"Added endpoints: {[route.path for route in app.routes]}")

    return app


def _add_pipeline_endpoint(
    app: FastAPI,
    endpoint_config: EndpointConfig,
    pipeline: Pipeline,
    integration: str = INTEGRATION_LOCAL,
):
    input_schema = pipeline.input_schema
    output_schema = pipeline.output_schema

    def _predict_from_schema(request: pipeline.input_schema):
        return pipeline(request)

    def _predict_from_files(request: List[UploadFile]):
        request = pipeline.input_schema.from_files(
            (file.file for file in request), from_server=True
        )
        return pipeline(request)

    routes_and_fns = []
    if integration == INTEGRATION_LOCAL:
        route = endpoint_config.route or "/predict"
        if not route.startswith("/"):
            route = "/" + route

        routes_and_fns.append((route, _predict_from_schema))
        if hasattr(input_schema, "from_files"):
            routes_and_fns.append((route + "/files", _predict_from_files))
    elif integration == INTEGRATION_SAGEMAKER:
        route = "/invocations"
        if hasattr(input_schema, "from_files"):
            routes_and_fns.append((route, _predict_from_files))
        else:
            routes_and_fns.append((route, _predict_from_schema))

    for route, endpoint_fn in routes_and_fns:
        app.add_api_route(
            route,
            endpoint_fn,
            response_model=output_schema,
            methods=["POST"],
            tags=["predict"],
        )
        _LOGGER.info(f"Added '{route}' endpoint")
