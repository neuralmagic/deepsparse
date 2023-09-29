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
from abc import abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List

import yaml
from pydantic import BaseModel

import uvicorn
from deepsparse.engine import Context
from deepsparse.pipeline import Pipeline
from deepsparse.server.config import ServerConfig, SystemLoggingConfig
from deepsparse.server.config_hot_reloading import start_config_watcher
from deepsparse.server.helpers import (
    prep_outputs_for_serialization,
    server_logger_from_config,
)
from deepsparse.server.system_logging import (
    SystemLoggingMiddleware,
    log_system_information,
)
from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import HTTPException
from starlette.responses import RedirectResponse


_LOGGER = logging.getLogger(__name__)


class CheckReady(BaseModel):
    status: str = "OK"


class ModelMetaData(BaseModel):
    model_path: str


class ProxyPipeline:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline


class Server:
    def __init__(self, config_path: str):
        _LOGGER.info(f"config_path: {config_path}")

        with open(config_path) as fp:
            obj = yaml.safe_load(fp)

        self.config_path = config_path
        self.server_config = ServerConfig(**obj)
        _LOGGER.info(f"Using config: {repr(self.server_config)}")

        self.context = Context(
            num_cores=self.server_config.num_cores,
            num_streams=self.server_config.num_workers,
        )
        self.executor = ThreadPoolExecutor(max_workers=self.context.num_streams)
        self.server_logger = server_logger_from_config(self.server_config)

    def start_server(
        self,
        host: str = "0.0.0.0",
        port: int = 5543,
        log_level: str = "info",
        hot_reload_config: bool = False,
    ):
        """
        Starts a FastAPI server with uvicorn with the configuration specified.

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

        if hot_reload_config:
            _LOGGER.info(f"Watching {self.config_path} for changes.")
            _ = start_config_watcher(
                self.config_path, f"http://{host}:{port}/endpoints", 0.5
            )

        app = self._build_app()

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            log_config=log_config,
            # NOTE: only want to have 1 server process so models aren't being copied
            workers=1,
        )

    def _build_app(self) -> FastAPI:
        route_counts = Counter([cfg.route for cfg in self.server_config.endpoints])
        if route_counts[None] > 1:
            raise ValueError(
                "You must specify `route` for all endpoints if multiple endpoints "
                "are used."
            )

        for route, count in route_counts.items():
            if count > 1:
                raise ValueError(
                    f"{route} specified {count} times for multiple EndpoingConfig.route"
                )

        self._set_pytorch_num_threads()
        self._set_thread_pinning()

        app = self._base_routes()
        return self._add_routes(app)

    def _base_routes(self) -> FastAPI:
        app = FastAPI()
        app.add_middleware(
            SystemLoggingMiddleware,
            server_logger=self.server_logger,
            system_logging_config=self.server_config.system_logging,
        )

        @app.get("/", include_in_schema=False)
        def _home():
            return RedirectResponse("/docs")

        @app.get("/config", tags=["general"], response_model=ServerConfig)
        def _info():
            return self.server_config

        @app.get("/ping", tags=["health"], response_model=CheckReady)
        @app.get("/v2/health/ready", tags=["health"], response_model=CheckReady)
        @app.get("/v2/health/live", tags=["health"], response_model=CheckReady)
        def _check_health():
            return CheckReady(status="OK")

        @app.get("/v2", tags=["metadata", "server"], response_model=str)
        def _get_server_info():
            return "This is the deepsparse server. Hello!"

        return app

    def _set_pytorch_num_threads(self):
        if self.server_config.pytorch_num_threads is not None:
            try:
                import torch

                torch.set_num_threads(self.server_config.pytorch_num_threads)
                _LOGGER.info(
                    f"torch.set_num_threads({self.server_config.pytorch_num_threads})"
                )
            except ImportError:
                _LOGGER.debug(
                    "pytorch not installed, skipping pytorch_num_threads configuration"
                )

    def _set_thread_pinning(self):
        pinning = {"core": ("1", "0"), "numa": ("0", "1"), "none": ("0", "0")}

        if self.server_config.engine_thread_pinning not in pinning:
            raise ValueError(
                "Invalid value for engine_thread_pinning. "
                'Expected one of {"core","numa","none"}. Found '
                f"{self.server_config.engine_thread_pinning}"
            )

        cores, socks = pinning[self.server_config.engine_thread_pinning]
        os.environ["NM_BIND_THREADS_TO_CORES"] = cores
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = socks

        _LOGGER.info(f"NM_BIND_THREADS_TO_CORES={cores}")
        _LOGGER.info(f"NM_BIND_THREADS_TO_SOCKETS={socks}")

    @abstractmethod
    def _add_routes(self, app):
        pass

    @staticmethod
    def pipeline_ready():
        return CheckReady(status="OK")

    @staticmethod
    def model_metadata(proxy_pipeline: ProxyPipeline):
        if not proxy_pipeline or not proxy_pipeline.pipeline.model_path:
            HTTPException(status_code=404, detail="Model path not found")
        return ModelMetaData(model_path=proxy_pipeline.pipeline.model_path)

    @staticmethod
    async def predict(
        proxy_pipeline: ProxyPipeline,
        system_logging_config: SystemLoggingConfig,
        raw_request: Request,
    ):
        request = proxy_pipeline.pipeline.input_schema(**await raw_request.json())
        pipeline_outputs = proxy_pipeline.pipeline(request)
        server_logger = proxy_pipeline.pipeline.logger
        if server_logger:
            log_system_information(
                server_logger=server_logger, system_logging_config=system_logging_config
            )
        pipeline_outputs = prep_outputs_for_serialization(pipeline_outputs)
        return pipeline_outputs

    @staticmethod
    def predict_from_files(proxy_pipeline: ProxyPipeline, request: List[UploadFile]):
        request = proxy_pipeline.pipeline.input_schema.from_files(
            (file.file for file in request), from_server=True
        )
        return Server.predict(request)
