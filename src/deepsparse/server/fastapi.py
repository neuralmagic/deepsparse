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

"""

"""

import logging
from pathlib import Path

import click

from deepsparse.server.asynchronous import execute_async
from deepsparse.server.config import (
    ServerConfig,
    server_config_from_env,
    server_config_to_env,
)
from deepsparse.server.utils import serializable_response
from deepsparse.version import version


try:
    import uvicorn
    from fastapi import FastAPI
    from starlette.responses import RedirectResponse
except Exception as err:
    raise ImportError(
        "Exception while importing install dependencies, "
        "run `pip install deepsparse[server]` to install the dependencies. "
        f"Recorded exception: {err}"
    )


ENV_DEEPSPARSE_SERVER_CONFIG = "DEEPSPARSE_SERVER_CONFIG"
_LOGGER = logging.getLogger(__name__)


def _add_general_routes(app, config):
    @app.get("/config", tags=["general"], response_model=ServerConfig)
    def _info():
        return config

    @app.get("/ping", tags=["general"])
    @app.get("/health", tags=["general"])
    @app.get("/healthcheck", tags=["general"])
    def _health():
        return {"status": "healthy"}

    @app.get("/", include_in_schema=False)
    def _home():
        return RedirectResponse("/docs")

    _LOGGER.info("created general routes, visit `/docs` to view available")


def _add_pipeline_route(app, pipeline_def, num_models: int, defined_tasks: set):
    async def _predict_func(request: pipeline_def.request_model):
        results = await execute_async(
            pipeline_def.pipeline, **vars(request), **pipeline_def.kwargs
        )
        return serializable_response(results)

    if num_models == 1:
        # add only a single predict route since this is the only model we're serving
        app.post(
            f"/predict", response_model=pipeline_def.response_model, tags=["prediction"]
        )(_predict_func)
    elif pipeline_def.config.alias:
        # add the prediction path under the given alias
        app.post(
            f"/predict/{pipeline_def.config.alias}",
            response_model=pipeline_def.response_model,
            tags=["prediction"],
        )(_predict_func)
    elif pipeline_def.config.task not in defined_tasks:
        # fall back on adding the model to the task provided nothing is already assigned
        app.post(
            f"/predict/{pipeline_def.config.task}",
            response_model=pipeline_def.response_model,
            tags=["prediction"],
        )(_predict_func)
        defined_tasks.add(pipeline_def.config.task)
    else:
        raise ValueError(
            f"Multiple tasks defined for {pipeline_def.config.task} and no alias "
            f"given for {pipeline_def.config}. "
            "Either define an alias or supply a single model for the task"
        )


def _server_app_factory():
    app = FastAPI(
        title="deepsparse.server",
        version=version,
        description="DeepSparse Inference Server",
    )
    _LOGGER.info("created FastAPI app for inference serving")

    config = server_config_from_env()
    _LOGGER.debug("loaded server config %s", config)
    _add_general_routes(app, config)

    pipeline_defs = load_pipelines_definitions(config)
    _LOGGER.debug("loaded pipeline definitions from config %s", pipeline_defs)
    num_tasks = len(config.models)
    defined_tasks = set()
    for pipeline_def in pipeline_defs:
        _add_pipeline_route(app, pipeline_def, num_tasks, defined_tasks)

    return app


@click.command()
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help=(
        "Bind socket to this host. Use --host 0.0.0.0 to make the application "
        "available on your local network. IPv6 addresses are supported, "
        "for example: --host '::'. Defaults to 0.0.0.0",
    ),
)
@click.option(
    "--port",
    type=str,
    default="5543",
    help="Bind to a socket with this port. Defaults to 5543.",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Use multiple worker processes. Defaults to 1.",
)
@click.option(
    "--log_level",
    type=str,
    default="5543",
    help="Bind to a socket with this port. Defaults to 5543.",
)
@click.option(
    "--config_file",
    type=str,
    default=None,
    help=(
        "Configuration file containing info on how to serve the desired models. "
        "See deepsparse.server.fastapi file for an example",
    ),
)
@click.option(
    "--task",
    type=str,
    default=None,
    help=(
        "The task the model_path is serving. For example, one of: "
        "question_answering, text_classification, token_classification. "
        "Ignored if config file is supplied",
    ),
)
@click.option(
    "--model_path",
    type=str,
    default=None,
    help=(
        "The path to a model.onnx file, a model folder containing the model.onnx "
        "and supporting files, or a SparseZoo model stub. "
        "Ignored if config_file is supplied.",
    ),
)
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help=(
        "The batch size to serve the model from model_path with."
        "Ignored if config_file is supplied.",
    ),
)
def start_server(
    host: str,
    port: str,
    workers: int,
    log_level: str,
    config_file: str,
    task: str,
    model_path: str,
    batch_size: int,
):
    _LOGGER.setLevel(log_level)
    server_config_to_env(config_file, task, model_path, batch_size)
    filename = Path(__file__).stem
    package = "deepsparse.transformers.server"
    app_name = f"{package}.{filename}:app"
    uvicorn.run(
        app_name,
        host=host,
        port=port,
        workers=workers,
        factory=True,
    )


if __name__ == "__main__":
    start_server()
