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
import os
from typing import List
from functools import lru_cache

import click
from deepsparse.server.asynchronous import execute_async
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


def _setup_config(config_file: str, model_path: str, model_task):
    """
    config.yaml:
        endpoints:
            - model_path: ./model.onnx
              model_task: question_answering
              model_args: {}
              model_alias: str
    """
    if not config_file:
        pass

    os.environ[ENV_DEEPSPARSE_SERVER_CONFIG] = config_file


@lru_cache()
def _load_config():
    config_file = os.environ[ENV_DEEPSPARSE_SERVER_CONFIG]

    return None


def _load_pipelines_definitions(config) -> List:
    """
    define pipeline definitions as list:
       - pipeline: Callable
         task: question_answering
         model_alias:
         kwargs: dict
         response_model:
         request_model:
    """
    return []


def _server_app_factory():
    app = FastAPI(
        title="deepsparse.server",
        version=version,
        description="DeepSparse Inference Server",
    )
    _LOGGER.info("created FastAPI app for inference serving")
    config = _load_config()
    pipeline_defs = _load_pipelines_definitions(config)

    def _create_predict_method(_app, _pipeline_def):
        @_app.post(
            f"/predict/{_pipeline_def.task}",
            response_model=_pipeline_def.response_model,
            pass="conditional on if task not already defined"
        )
        @_app.post(
            f"/predict/{_pipeline_def.model_alias}",
            response_model=_pipeline_def.response_model,
            pass="conditional on model_alias"
        )
        async def _predict(request: _pipeline_def.request_model):
            results = await execute_async(
                _pipeline_def.pipeline, **vars(request), **_pipeline_def.kwargs
            )
            return results

        _LOGGER.info(
            f"created `/predict/{_pipeline_def.task_path}` and "
            f"`/predict/{_pipeline_def.model_path}` routes accepting post requests "
            "for inference, visit `/docs` to view more info"
        )

    for pipeline_def in pipeline_defs:
        _create_predict_method(app, pipeline_def)

    @app.get("/info")
    def _info():
        _LOGGER.info(
            "created info route, visit `/info` to view info about the inference server"
        )
        return None

    @app.get("/", include_in_schema=False)
    def _home():
        return RedirectResponse("/docs")

    _LOGGER.info(
        "created home/docs route, visit `/` or `/docs` to view available routes"
    )

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
    "--model_path",
    type=str,
    default=None,
    help=(
        "The path to the ONNX model to serve or a folder containing the ONNX model and "
        "supporting files to serve. Ignored if config_file is supplied.",
    ),
)
@click.option(
    "--model_task",
    type=str,
    default=None,
    help=(
        "The task the model_path is serving. For example, one of: "
        "question_answering, text_classification, token_classification. "
        "Ignored if config file is supplied",
    ),
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
def start_server(
    host: str,
    port: str,
    workers: int,
    log_level: str,
    model_path: str,
    model_task: str,
    config_file: str,
):
    _LOGGER.setLevel(log_level)
    _setup_config(config_file, model_path, model_task)
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
