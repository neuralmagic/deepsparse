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
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import yaml

import uvicorn
from deepsparse.engine import Context, Scheduler
from deepsparse.pipeline import (
    DEEPSPARSE_ENGINE,
    Pipeline,
    PipelineConfig,
    SupportedTasks,
)
from deepsparse.server_v2.config import (
    EndpointConfig,
    ImageSizesConfig,
    SequenceLengthsConfig,
    ServerConfig,
)
from fastapi import FastAPI, UploadFile
from starlette.responses import RedirectResponse


_LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=5543)
@click.option(
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warn", "critical", "fatal"], case_sensitive=False
    ),
    default="info",
    help="Sets the logging level. Defaults to info.",
)
@click.argument("config-path", type=str)
def main(config_path: str, host: str, port: int, log_level: str):
    start_server(config_path, host, port, log_level)


def start_server(
    config_path: str, host: str = "0.0.0.0", port: int = 5543, log_level: str = "info"
):
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
        num_streams=server_config.num_concurrent_batches,
        scheduler=_get_scheduler(server_config),
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

    for model_config in server_config.endpoints:
        pipeline_config = _model_config_to_pipeline_config(model_config)
        pipeline_config.kwargs["executor"] = app._deepsparse_pool

        _LOGGER.info(f"Initializing pipeline for '{model_config.name}'")
        pipeline = Pipeline.from_config(pipeline_config, context)

        _LOGGER.info(f"Adding endpoints for '{model_config.name}'")
        _add_model_endpoint(app, model_config, pipeline, add_invocations_endpoint)

    _LOGGER.info(f"Added endpoints: {[route.path for route in app.routes]}")

    return app


def _add_model_endpoint(
    app: FastAPI,
    model_config: EndpointConfig,
    pipeline: Pipeline,
    add_invocations_endpoint: bool = False,
):
    input_schema = pipeline.input_schema
    output_schema = pipeline.output_schema

    pool: ThreadPoolExecutor = app._deepsparse_pool

    @app.post(model_config.endpoint, tags=["predict"], response_model=output_schema)
    async def _predict_func(request: pipeline.input_schema):
        return await get_running_loop().run_in_executor(pool, pipeline, request)

    _LOGGER.info(f"Added '{model_config.endpoint}' endpoint")

    if hasattr(input_schema, "from_files"):
        file_endpoint = model_config.endpoint + "/files"

        @app.post(file_endpoint, tags=["predict"], response_model=output_schema)
        async def _predict_from_files_func(request: List[UploadFile]):
            request = pipeline.input_schema.from_files(
                (file.file for file in request), from_server=True
            )
            return await get_running_loop().run_in_executor(pool, pipeline, request)

        _LOGGER.info(f"Added '{file_endpoint}' endpoint")

    if add_invocations_endpoint and hasattr(input_schema, "from_files"):

        @app.post("/invocations", response_model=output_schema)
        async def _invocations(request: List[UploadFile]):
            return RedirectResponse(file_endpoint)

        _LOGGER.info("Added '/invocations' endpoint")


def _get_scheduler(server_config: ServerConfig) -> Scheduler:
    return (
        Scheduler.multi_stream
        if server_config.num_concurrent_batches > 1
        else Scheduler.single_stream
    )


def _model_config_to_pipeline_config(model: EndpointConfig) -> PipelineConfig:
    if model.batch_size == 1 and model.accept_multiples_of_batch_size:
        # dynamic batch
        batch_size = None
    elif model.batch_size > 1 and model.accept_multiples_of_batch_size:
        # should still be dynamic batch
        raise NotImplementedError
    else:
        batch_size = model.batch_size

    input_shapes, kwargs = _unpack_bucketing(model.task, model.bucketing)

    return PipelineConfig(
        task=model.task,
        model_path=model.model,
        engine_type=DEEPSPARSE_ENGINE,
        batch_size=batch_size,
        num_cores=None,  # this will be set from Context
        input_shapes=input_shapes,
        kwargs=kwargs,
    )


def _unpack_bucketing(
    task: str, bucketing: Optional[Union[SequenceLengthsConfig, ImageSizesConfig]]
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """
    :return: (input_shapes, kwargs) which are passed to PipelineConfig
    """
    if bucketing is None:
        return None, {}

    if isinstance(bucketing, SequenceLengthsConfig):
        if not SupportedTasks.is_nlp(task):
            raise ValueError(f"SequenceLengthConfig specified for non-nlp task {task}")

        return _unpack_nlp_bucketing(bucketing)
    elif isinstance(bucketing, ImageSizesConfig):
        if (
            not SupportedTasks.is_image_classification(task)
            and not SupportedTasks.is_yolo(task)
            and not SupportedTasks.is_yolact(task)
        ):
            raise ValueError(
                f"ImageSizeConfig specified for non computer vision task {task}"
            )

        return _unpack_cv_bucketing(bucketing)
    else:
        raise ValueError(f"Unknown bucket config {bucketing}")


def _unpack_nlp_bucketing(cfg: SequenceLengthsConfig):
    if len(cfg.sequence_lengths) == 0:
        raise ValueError("Must specify at least one sequence length under bucketing")

    if len(cfg.sequence_lengths) == 1:
        input_shapes = None
        kwargs = {"sequence_length": cfg.sequence_lengths[0]}
    else:
        input_shapes = None
        kwargs = {"sequence_length": cfg.sequence_lengths}

    return input_shapes, kwargs


def _unpack_cv_bucketing(cfg: ImageSizesConfig):
    if len(cfg.image_sizes) == 0:
        raise ValueError("Must specify at least one image size under bucketing")

    if len(cfg.image_sizes) == 1:
        # NOTE: convert from List[Tuple[int, int]] to List[List[int]]
        input_shapes = [list(cfg.image_sizes[0])]
        kwargs = {}
    else:
        raise NotImplementedError(
            "Multiple image size buckets is currently unsupported"
        )

    return input_shapes, kwargs


if __name__ == "__main__":
    main()
