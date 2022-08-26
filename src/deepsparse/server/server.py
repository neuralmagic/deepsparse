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
from asyncio import get_running_loop
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List

import yaml

import uvicorn
from deepsparse.engine import Context, Scheduler
from deepsparse.pipeline import Pipeline
from deepsparse.server.config import EndpointConfig, ServerConfig
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
    context = Context(
        num_cores=server_config.num_cores,
        num_streams=server_config.num_workers,
        scheduler=Scheduler.elastic,
    )

    _LOGGER.info(f"Built context: {repr(context)}")

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

    app._deepsparse_pool = ThreadPoolExecutor(max_workers=context.num_streams)

    add_invocations_endpoint = len(server_config.endpoints) == 1

    # fill in names if there are none
    for idx, endpoint_config in enumerate(server_config.endpoints):
        if endpoint_config.name is None:
            endpoint_config.name = f"endpoint-{idx}"

    # creat pipelines & endpoints
    for endpoint_config in server_config.endpoints:
        pipeline_config = endpoint_config.to_pipeline_config()
        pipeline_config.kwargs["executor"] = app._deepsparse_pool

        _LOGGER.info(f"Initializing pipeline for '{endpoint_config.name}'")
        pipeline = Pipeline.from_config(pipeline_config, context)

        _LOGGER.info(f"Adding endpoints for '{endpoint_config.name}'")
        _add_pipeline_endpoint(app, endpoint_config, pipeline, add_invocations_endpoint)

    _LOGGER.info(f"Added endpoints: {[route.path for route in app.routes]}")

    return app


def _add_pipeline_endpoint(
    app: FastAPI,
    endpoint_config: EndpointConfig,
    pipeline: Pipeline,
    add_invocations_endpoint: bool = False,
):
    input_schema = pipeline.input_schema
    output_schema = pipeline.output_schema

    pool: ThreadPoolExecutor = app._deepsparse_pool

    route = endpoint_config.endpoint or "/predict"
    if not route.startswith("/"):
        route = "/" + route

    @app.post(route, tags=["predict"], response_model=output_schema)
    async def _predict_func(request: pipeline.input_schema):
        return await get_running_loop().run_in_executor(pool, pipeline, request)

    _LOGGER.info(f"Added '{route}' endpoint")

    if hasattr(input_schema, "from_files"):
        file_route = route + "/files"

        @app.post(file_route, tags=["predict"], response_model=output_schema)
        async def _predict_from_files_func(request: List[UploadFile]):
            request = pipeline.input_schema.from_files(
                (file.file for file in request), from_server=True
            )
            return await get_running_loop().run_in_executor(pool, pipeline, request)

        _LOGGER.info(f"Added '{file_route}' endpoint")

    if add_invocations_endpoint and hasattr(input_schema, "from_files"):

        @app.post("/invocations", response_model=output_schema)
        async def _invocations(request: List[UploadFile]):
            return RedirectResponse(file_route)

        _LOGGER.info("Added '/invocations' endpoint")
