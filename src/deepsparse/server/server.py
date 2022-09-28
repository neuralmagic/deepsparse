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
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List

import yaml

import uvicorn
from deepsparse.engine import Context
from deepsparse.loggers import ManagerLogger
from deepsparse.pipeline import Pipeline
from deepsparse.server.config import (
    INTEGRATION_LOCAL,
    INTEGRATION_SAGEMAKER,
    INTEGRATIONS,
    EndpointConfig,
    ServerConfig,
)
from deepsparse.server.config_hot_reloading import start_config_watcher
from deepsparse.server.helpers import default_logger_manager, logger_manager_from_config
from fastapi import FastAPI, UploadFile
from starlette.responses import RedirectResponse


_LOGGER = logging.getLogger(__name__)


def start_server(
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 5543,
    log_level: str = "info",
    hot_reload_config: bool = False,
):
    """
    Starts a FastAPI server with uvicorn with the configuration specified.

    :param config_path: A yaml file with the server config. See :class:`ServerConfig`.
    :param host: The IP address to bind the server to.
    :param port: The port to listen on.
    :param log_level: Log level given to python and uvicorn logging modules.
    :param hot_reload_config: `True` to reload the config file if it is modified.
    """
    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"][__name__] = {
        "handlers": ["default"],
        "level": log_level.upper(),
    }

    _LOGGER.info(f"config_path: {config_path}")

    with open(config_path) as fp:
        obj = yaml.safe_load(fp)
    server_config = ServerConfig(**obj)
    _LOGGER.info(f"Using config: {repr(server_config)}")

    if hot_reload_config:
        _LOGGER.info(f"Watching {config_path} for changes.")
        _ = start_config_watcher(config_path, f"http://{host}:{port}/endpoints", 0.5)

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

    _set_pytorch_num_threads(server_config)
    _set_thread_pinning(server_config)

    context = Context(
        num_cores=server_config.num_cores,
        num_streams=server_config.num_workers,
    )
    executor = ThreadPoolExecutor(max_workers=context.num_streams)

    _LOGGER.info(f"Built context: {repr(context)}")
    _LOGGER.info(f"Built ThreadPoolExecutor with {executor._max_workers} workers")

    app = FastAPI()
    pipeline_logger = _initialize_loggers(server_config)

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

    @app.post("/endpoints", tags=["endpoints"], response_model=bool)
    def _add_endpoint_endpoint(cfg: EndpointConfig):
        if cfg.name is None:
            cfg.name = f"endpoint-{len(app.routes)}"
        _add_endpoint(app, server_config, cfg, executor, context, pipeline_logger)
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

    # fill in names if there are none
    for idx, endpoint_config in enumerate(server_config.endpoints):
        if endpoint_config.name is None:
            endpoint_config.name = f"endpoint-{idx}"

    # create pipelines & endpoints
    for endpoint_config in server_config.endpoints:
        _add_endpoint(
            app, server_config, endpoint_config, executor, context, pipeline_logger
        )

    _LOGGER.info(f"Added endpoints: {[route.path for route in app.routes]}")

    return app


def _set_pytorch_num_threads(server_config: ServerConfig):
    if server_config.pytorch_num_threads is not None:
        try:
            import torch

            torch.set_num_threads(server_config.pytorch_num_threads)
            _LOGGER.info(f"torch.set_num_threads({server_config.pytorch_num_threads})")
        except ImportError:
            _LOGGER.debug(
                "pytorch not installed, skipping pytorch_num_threads configuration"
            )


def _set_thread_pinning(server_config: ServerConfig):
    pinning = {"core": ("1", "0"), "numa": ("0", "1"), "none": ("0", "0")}

    if server_config.engine_thread_pinning not in pinning:
        raise ValueError(
            "Invalid value for engine_thread_pinning. "
            'Expected one of {"core","numa","none"}. Found '
            f"{server_config.engine_thread_pinning}"
        )

    cores, socks = pinning[server_config.engine_thread_pinning]
    os.environ["NM_BIND_THREADS_TO_CORES"] = cores
    os.environ["NM_BIND_THREADS_TO_SOCKETS"] = socks

    _LOGGER.info(f"NM_BIND_THREADS_TO_CORES={cores}")
    _LOGGER.info(f"NM_BIND_THREADS_TO_SOCKETS={socks}")


def _add_endpoint(
    app: FastAPI,
    server_config: ServerConfig,
    endpoint_config: EndpointConfig,
    executor: ThreadPoolExecutor,
    context: Context,
    pipeline_logger: ManagerLogger,
):
    pipeline_config = endpoint_config.to_pipeline_config()
    pipeline_config.kwargs["executor"] = executor

    _LOGGER.info(f"Initializing pipeline for '{endpoint_config.name}'")
    pipeline = Pipeline.from_config(pipeline_config, context)

    _LOGGER.info(f"Adding endpoints for '{endpoint_config.name}'")
    _add_pipeline_endpoint(
        app, endpoint_config, pipeline, pipeline_logger, server_config.integration
    )


def _add_pipeline_endpoint(
    app: FastAPI,
    endpoint_config: EndpointConfig,
    pipeline: Pipeline,
    pipeline_logger: ManagerLogger,
    integration: str = INTEGRATION_LOCAL,
):
    input_schema = pipeline.input_schema
    output_schema = pipeline.output_schema
    pipeline_name = pipeline.alias or pipeline.task

    def _predict(request: pipeline.input_schema):
        (
            pipeline_outputs,
            pipeline_inputs,
            engine_inputs,
            inference_timing,
        ) = pipeline.run_with_monitoring(request)
        pipeline_logger.log_latency(pipeline_name, inference_timing)
        pipeline_logger.log_data(
            pipeline_name, (pipeline_inputs, engine_inputs), pipeline_outputs
        )
        return pipeline_outputs

    def _predict_from_files(request: List[UploadFile]):
        request = pipeline.input_schema.from_files(
            (file.file for file in request), from_server=True
        )
        return _predict(request)

    routes_and_fns = []
    if integration == INTEGRATION_LOCAL:
        route = endpoint_config.route or "/predict"
        if not route.startswith("/"):
            route = "/" + route

        routes_and_fns.append((route, _predict))
        if hasattr(input_schema, "from_files"):
            routes_and_fns.append((route + "/from_files", _predict_from_files))
    elif integration == INTEGRATION_SAGEMAKER:
        route = "/invocations"
        if hasattr(input_schema, "from_files"):
            routes_and_fns.append((route, _predict_from_files))
        else:
            routes_and_fns.append((route, _predict))

    for route, endpoint_fn in routes_and_fns:
        app.add_api_route(
            route,
            endpoint_fn,
            response_model=output_schema,
            methods=["POST"],
            tags=["predict"],
        )
        _LOGGER.info(f"Added '{route}' endpoint")


def _initialize_loggers(server_config: ServerConfig) -> ManagerLogger:
    loggers_config = server_config.loggers
    if loggers_config is None:
        return ManagerLogger([])
    if isinstance(loggers_config, str):
        if not loggers_config == "default":
            raise ValueError(
                f"given string {loggers_config} for ServerConfig.loggers only "
                "supported string is 'default', other configs should be specified "
                "with a dict literal of logging integration to their initialization "
                "kwargs"
            )
        else:
            return default_logger_manager()
    return logger_manager_from_config(server_config.loggers)
