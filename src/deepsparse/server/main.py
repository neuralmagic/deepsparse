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

# flake8: noqa

"""
Serving script for ONNX models and configurations with the DeepSparse engine.

##########
Command help:
deepsparse.server --help
Usage: deepsparse.server [OPTIONS]

  Start a DeepSparse inference server for serving the models and pipelines
  given within the config_file or a single model defined by task, model_path,
  and batch_size.

  Example config.yaml for serving:

  models:
      - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
        batch_size: 1
        alias: question_answering/dense
      - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95
        batch_size: 1
        alias: question_answering/sparse_quantized

Options:
  --host TEXT           Bind socket to this host. Use --host 0.0.0.0 to make
                        the application available on your local network. IPv6
                        addresses are supported, for example: --host '::'.
                        Defaults to 0.0.0.0.
  --port INTEGER        Bind to a socket with this port. Defaults to 5543.
  --workers INTEGER     Use multiple worker processes. Defaults to 1.
  --log_level TEXT      Sets the logging level. Defaults to info.
  --config_file TEXT    Configuration file containing info on how to serve the
                        desired models.
  --task [custom|question_answering|qa|text_classification|glue|sentiment_analysis|
  token_classification|ner|zero_shot_text_classification|embedding_extraction|
  image_classification|yolo|yolact|information_retrieval_haystack|haystack]
                        The task the model_path is serving.
  --model_path TEXT     The path to a model.onnx file, a model folder
                        containing the model.onnx and supporting files, or a
                        SparseZoo model stub. Ignored if config_file is
                        supplied.
  --batch_size INTEGER  The batch size to serve the model from model_path
                        with. Ignored if config_file is supplied.
  --integration [default|sagemaker]
                                  Name of deployment integration that this
                                  server will be deployed to Currently
                                  supported options are 'default' and
                                  'sagemaker' for inference deployment with
                                  Amazon Sagemaker
  --help                Show this message and exit.


##########
Example for serving a single BERT model from the SparseZoo
deepsparse.server \
    --task question_answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"

##########
Example for serving models from a config file
deepsparse.server \
    --config_file config.yaml
"""

import logging
from pathlib import Path
from typing import List

import click

from deepsparse import Context, Pipeline
from deepsparse.log import set_logging_level
from deepsparse.server.asynchronous import execute_async, initialize_async
from deepsparse.server.config import (
    ServerConfig,
    server_config_from_env,
    server_config_to_env,
)
from deepsparse.server.utils import serializable_response
from deepsparse.tasks import SupportedTasks
from deepsparse.version import version


try:
    import uvicorn
    from fastapi import FastAPI, UploadFile
    from starlette.responses import RedirectResponse
except Exception as err:
    raise ImportError(
        "Exception while importing install dependencies, "
        "run `pip install deepsparse[server]` to install the dependencies. "
        f"Recorded exception: {err}"
    )


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


def _add_pipeline_route(
    app,
    pipeline: Pipeline,
    num_models: int,
    defined_tasks: set,
    integration: str,
):
    def _create_endpoint(endpoint_path: str, from_files: bool = False):
        # if `from_files` is True, the endpoint expects request to be
        # `List[UploadFile]` otherwise, the endpoint expect request to
        # be `pipeline.input_schema`
        input_schema = List[UploadFile] if from_files else pipeline.input_schema

        @app.post(
            endpoint_path,
            response_model=pipeline.output_schema,
            tags=["prediction"],
        )
        async def _predict_func(request: input_schema):
            if from_files:
                request = pipeline.input_schema.from_files(
                    file.file for file in request
                )

            results = await execute_async(
                pipeline,
                request,
            )
            return serializable_response(results)

        _LOGGER.info(f"created route {endpoint_path}")

        return _predict_func

    path = "/predict"

    if integration.lower() == "sagemaker":
        if num_models > 1:
            raise ValueError(
                "Sagemaker inference with deepsparse.server currently supports "
                f"serving one model, received config for {num_models} models"
            )
        # required path name for Sagemaker
        path = "/invocations"
    elif pipeline.alias:
        path = f"/predict/{pipeline.alias}"
    elif num_models > 1:
        if pipeline.task in defined_tasks:
            raise ValueError(
                f"Multiple tasks defined for {pipeline.task} and no alias "
                f"given for pipeline with model {pipeline.model_path_orig}. "
                "Either define an alias or supply a single model for the task"
            )
        path = f"/predict/{pipeline.task}"
        defined_tasks.add(pipeline.task)

    if hasattr(pipeline.input_schema, "from_files"):
        if integration.lower() == "sagemaker":
            # SageMaker supports one endpoint per model, using file upload path
            _create_endpoint(path, from_files=True)
        else:
            # create endpoint for json and file input
            _create_endpoint(path)
            _create_endpoint(path + "/from_files", from_files=True)
    else:
        # create endpoint with no file support
        _create_endpoint(path)


def server_app_factory():
    """
    :return: a FastAPI app initialized with defined routes and ready for inference.
        Use with a wsgi server such as uvicorn.
    """
    app = FastAPI(
        title="deepsparse.server",
        version=version,
        description="DeepSparse Inference Server",
        swagger_ui_parameters={"syntaxHighlight": False},
    )
    _LOGGER.info("created FastAPI app for inference serving")

    config = server_config_from_env()
    initialize_async(config.workers)
    _LOGGER.debug("loaded server config %s", config)
    _add_general_routes(app, config)

    num_cores = None
    for model_config in config.models:
        num_cores = (
            max(num_cores, model_config.num_cores)
            if num_cores is not None and model_config.num_cores is not None
            else num_cores or model_config.num_cores
        )

    context = Context(num_cores=num_cores)
    pipelines = [
        Pipeline.from_config(model_config, context=context)
        for model_config in config.models
    ]
    _LOGGER.debug("loaded pipeline definitions from config %s", pipelines)
    num_tasks = len(config.models)
    defined_tasks = set()
    for pipeline in pipelines:
        _add_pipeline_route(app, pipeline, num_tasks, defined_tasks, config.integration)

    return app


@click.command(
    context_settings=dict(token_normalize_func=lambda x: x.replace("-", "_"))
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Bind socket to this host. Use --host 0.0.0.0 to make the application "
    "available on your local network. "
    "IPv6 addresses are supported, for example: --host '::'. Defaults to 0.0.0.0",
)
@click.option(
    "--port",
    type=int,
    default=5543,
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
    type=click.Choice(
        ["debug", "info", "warn", "critical", "fatal"], case_sensitive=False
    ),
    default="info",
    help="Sets the logging level. Defaults to info.",
)
@click.option(
    "--config_file",
    type=str,
    default=None,
    help="Configuration file containing info on how to serve the desired models.",
)
@click.option(
    "--task",
    type=click.Choice(SupportedTasks.task_names(), case_sensitive=False),
    default=None,
    help="The task the model_path is serving.",
)
@click.option(
    "--model_path",
    type=str,
    default=None,
    help="The path to a model.onnx file, a model folder containing the model.onnx "
    "and supporting files, or a SparseZoo model stub. "
    "Ignored if config_file is supplied.",
)
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help="The batch size to serve the model from model_path with. "
    "Ignored if config_file is supplied.",
)
@click.option(
    "--integration",
    type=click.Choice(["default", "sagemaker"], case_sensitive=False),
    default="default",
    help="Name of deployment integration that this server will be deployed to "
    "Currently supported options are 'default' and 'sagemaker' for "
    "inference deployment with Amazon Sagemaker",
)
def start_server(
    host: str,
    port: int,
    workers: int,
    log_level: str,
    config_file: str,
    task: str,
    model_path: str,
    batch_size: int,
    integration: str,
):
    """
    Start a DeepSparse inference server for serving the models and pipelines given
    within the config_file or a single model defined by task, model_path, and batch_size

    Example config.yaml for serving:

    \b
    models:
        - task: question_answering
          model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
          batch_size: 1
          alias: question_answering/dense
        - task: question_answering
          model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95
          batch_size: 1
          alias: question_answering/sparse_quantized
    """
    set_logging_level(getattr(logging, log_level.upper()))
    server_config_to_env(config_file, task, model_path, batch_size, integration)
    filename = Path(__file__).stem
    package = "deepsparse.server"
    app_name = f"{package}.{filename}:server_app_factory"
    uvicorn.run(
        app_name,
        host=host,
        port=port,
        workers=workers,
        factory=True,
    )


if __name__ == "__main__":
    start_server()
