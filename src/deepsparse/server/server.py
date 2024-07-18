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
from copy import deepcopy
from typing import AsyncGenerator, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict

import uvicorn
from deepsparse.benchmark.benchmark_pipeline import benchmark_from_pipeline
from deepsparse.benchmark.config import PipelineBenchmarkConfig
from deepsparse.engine import Context
from deepsparse.pipeline import Pipeline
from deepsparse.server.config import ServerConfig, SystemLoggingConfig
from deepsparse.server.config_hot_reloading import start_config_watcher
from deepsparse.server.helpers import server_logger_from_config
from deepsparse.server.system_logging import (
    SystemLoggingMiddleware,
    log_system_information,
)
from deepsparse.utils import InferenceState
from deepsparse.utils.data import prep_for_serialization
from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse


_LOGGER = logging.getLogger(__name__)


class CheckReady(BaseModel):
    status: str = "OK"


class ModelMetaData(BaseModel):
    model_path: str

    # override name spaces due to model_ warnings in pydantic 2.X
    model_config = ConfigDict(protected_namespaces=())


# For deepsparse endpoints, we bind the `predict`` and `predict_from_files` functions to
# each of the added routes. As we require access to the pipeline to run inference on
# each request, instead of binding `predict`, we bind `partial(predict, pipeline ...)`
# so that there is access to the pipelne when handling each request. However, fastapi
# has trouble with validating the pipeline type. As a workaround, we can wrap each
# pipelinne as a `ProxyPipeline` which can be resolved by fastapi.
class ProxyPipeline:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline


class Server:
    def __init__(self, server_config: Union[str, ServerConfig]):

        if isinstance(server_config, str):
            with open(server_config) as fp:
                obj = yaml.safe_load(fp)

            self.config_path = server_config
            self.server_config = ServerConfig(**obj)
        else:
            self.server_config = server_config

        _LOGGER.info(f"Using config: {repr(self.server_config)}")
        self.server_logger = server_logger_from_config(self.server_config)
        self.context = Context(
            num_cores=self.server_config.num_cores,
            num_streams=self.server_config.num_workers,
        )

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

        for route, count in route_counts.items():
            if count > 1 and route is not None:
                raise ValueError(
                    f"{route} specified {count} times for multiple EndpoingConfig.route"
                )

        self._set_pytorch_num_threads()
        self._set_thread_pinning()

        app = self._base_routes()
        return self._add_routes(app)

    def _base_routes(self) -> FastAPI:
        app = FastAPI(title="DeepSparse Server API")
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

    def clean_up_route(self, route):
        if not route.startswith("/"):
            route = "/" + route
        if route.endswith("/"):
            route = route[:-1]
        return route

    def _update_routes(
        self,
        app: FastAPI,
        routes_and_fns: List,
        methods: List[str],
        tags: List[str],
        response_model: Optional[BaseModel] = None,
    ):
        existing = [route.path for route in app.routes]
        for route, endpoint_fn in routes_and_fns:
            if route not in existing:
                app.add_api_route(
                    route,
                    endpoint_fn,
                    response_model=response_model,
                    methods=methods,
                    tags=tags,
                )
                _LOGGER.info(f"Added '{route}' endpoint")

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

        if hasattr(proxy_pipeline.pipeline, "run_async"):
            inference_state = InferenceState()
            inference_state.create_state({})

            pipeline_outputs = await proxy_pipeline.pipeline.run_async(
                **await raw_request.json(), inference_state=inference_state
            )

            if isinstance(pipeline_outputs, AsyncGenerator):

                async def format_response():
                    async for output in pipeline_outputs:
                        # Note: If a pipeline does not return a BaseModel this will fail
                        response = output.json(ensure_ascii=False)
                        yield f"{response}\n\n"

                return StreamingResponse(
                    format_response(),
                    media_type="text/event-stream",
                )
        else:
            pipeline_outputs = proxy_pipeline.pipeline(**await raw_request.json())

        try:
            server_logger = proxy_pipeline.pipeline.logger
            if server_logger:
                log_system_information(
                    server_logger=server_logger,
                    system_logging_config=system_logging_config,
                )
        except Exception as e:
            _LOGGER.debug(f"Logging failed, {e}")

        return prep_for_serialization(pipeline_outputs)

    @staticmethod
    async def benchmark(
        proxy_pipeline: ProxyPipeline,
        system_logging_config: SystemLoggingConfig,
        raw_request: Request,
    ):
        json_params = await raw_request.json()
        benchmark_config = PipelineBenchmarkConfig(**json_params)
        results = benchmark_from_pipeline(
            pipeline=proxy_pipeline.pipeline, **benchmark_config.model_dump()
        )

        return results

    @staticmethod
    async def predict_from_files(
        proxy_pipeline: ProxyPipeline,
        system_logging_config: SystemLoggingConfig,
        request: List[UploadFile],
    ):
        # TODO: New pipelines do not have to have an input_schema.
        # enable from_files for image_classification for now; leverage SupportedTasks
        # or additional pipeline attributes to do this in the future
        if hasattr(proxy_pipeline.pipeline, "run_async"):
            inference_state = InferenceState()
            inference_state.create_state({})

            request = proxy_pipeline.pipeline.ops[0].input_schema.from_files(
                (file.file for file in request), from_server=True
            )
            pipeline_outputs = await proxy_pipeline.pipeline.run_async(
                request, inference_state=InferenceState
            )
        else:
            request = proxy_pipeline.pipeline.input_schema.from_files(
                (file.file for file in request), from_server=True
            )
            pipeline_outputs = proxy_pipeline.pipeline(request)

        try:
            server_logger = proxy_pipeline.pipeline.logger
            if server_logger:
                log_system_information(
                    server_logger=server_logger,
                    system_logging_config=system_logging_config,
                )
        except Exception as e:
            _LOGGER.debug(f"Logging failed, {e}")
        return prep_for_serialization(pipeline_outputs)
