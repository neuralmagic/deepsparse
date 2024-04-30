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
import os
from tempfile import TemporaryDirectory
from typing import Optional, Union

import click
import yaml

# TODO: update to use new tasks once server support lands
from deepsparse.legacy.tasks import SupportedTasks
from deepsparse.server.config import (
    INTEGRATION_LOCAL,
    INTEGRATION_OPENAI,
    INTEGRATION_SAGEMAKER,
    INTEGRATIONS,
    EndpointConfig,
    ServerConfig,
)
from deepsparse.server.deepsparse_server import DeepsparseServer
from deepsparse.server.openai_server import OpenAIServer
from deepsparse.server.sagemaker import SagemakerServer


HOST_OPTION = click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help=(
        "Bind socket to this host. Use --host 0.0.0.0 to make the application "
        "available on your local network. "
        "IPv6 addresses are supported, for example: --host '::'."
    ),
)

PORT_OPTION = click.option(
    "--port",
    type=int,
    default=5543,
    help="Bind to a socket with this port.",
)

LOG_LEVEL_OPTION = click.option(
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warn", "critical", "fatal"], case_sensitive=False
    ),
    default="info",
    help="Sets the logging level.",
)

HOT_RELOAD_OPTION = click.option(
    "--hot-reload-config",
    is_flag=True,
    default=False,
    help=(
        "Hot reload the config whenever the file is updated."
        "Deployed endpoints will be updated based on latest config."
    ),
)

MODEL_ARG = click.argument("model", type=str, default=None, required=False)
MODEL_OPTION = click.option(
    "--model_path",
    type=str,
    default="default",
    help=(
        "The path to a model.onnx file, a model folder containing the model.onnx "
        "and supporting files, or a SparseZoo model stub. "
        "If not specified, the default model for the task is used."
    ),
)

BATCH_OPTION = click.option(
    "--batch_size",
    type=int,
    default=1,
    help="The batch size to serve the model from model_path with",
)

CORES_OPTION = click.option(
    "--num-cores",
    type=int,
    default=None,
    help=(
        "The number of cores available for model execution. "
        "Defaults to all available cores."
    ),
)

WORKERS_OPTION = click.option(
    "--num-workers",
    type=int,
    default=None,
    help=(
        "The number of workers to split the available cores between. "
        "Defaults to half of the num_cores set"
    ),
)

INTEGRATION_OPTION = click.option(
    "--integration",
    type=click.Choice(INTEGRATIONS, case_sensitive=False),
    default="local",
    help=(
        "Name of deployment integration that this server will be deployed to "
        "Currently supported options are 'default', 'openai', or 'sagemaker' for "
        "inference deployment with Amazon Sagemaker"
    ),
)


@click.group(
    invoke_without_command=True,
    context_settings=dict(
        token_normalize_func=lambda x: "_".join(x.replace("-", "_").split()),
        show_default=True,
    ),
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
@HOST_OPTION
@PORT_OPTION
@LOG_LEVEL_OPTION
@HOT_RELOAD_OPTION
@MODEL_ARG
@MODEL_OPTION
@BATCH_OPTION
@CORES_OPTION
@WORKERS_OPTION
@INTEGRATION_OPTION
@click.pass_context
def main(
    ctx,
    task: Optional[str],
    config_file: Optional[str],
    host: str,
    port: int,
    log_level: str,
    hot_reload_config: bool,
    model_path: str,
    model: str,
    batch_size: int,
    num_cores: int,
    num_workers: int,
    integration: str,
):
    """
    Start a DeepSparse inference server for serving the models and pipelines.

        1. `deepsparse.server --config_file <config path> [OPTIONS]`

        2. `deepsparse.server --task <task> [OPTIONS]`

    Examples for using the server:

        `deepsparse.server --config_file server-config.yaml`

        `deepsparse.server --task question_answering --batch-size 2`

        `deepsparse.server --task question_answering --host "0.0.0.0"`

    Example config.yaml for serving:

    ```yaml
    num_cores: 2
    num_workers: 2
    endpoints:
        - task: question_answering
          route: /unpruned/predict
          model: zoo:some/zoo/stub
        - task: question_answering
          route: /pruned/predict
          model: /path/to/local/model
    ```

    To manually specify the set of loggers:

    ```yaml
     num_cores: 2
     num_workers: 2
     loggers:
     prometheus:
         port: 6100
         text_log_save_dir: /home/deepsparse-server/prometheus
         text_log_save_frequency: 30
     endpoints:
     - task: question_answering
       ...
    ```
    """
    # the server cli can take a model argument or --model_path option
    # if the --model_path option is provided, use that
    # otherwise if the argument is given and --model_path is not used, use the
    # argument instead

    if model and model_path == "default":
        model_path = model

    if integration == INTEGRATION_OPENAI:
        if task is None or task != "text_generation":
            task = "text_generation"

    if ctx.invoked_subcommand is not None:
        return

    if config_file is not None:
        server = _fetch_server(integration=integration, config=config_file)
        server.start_server(host, port, log_level, hot_reload_config=hot_reload_config)

    elif task is not None:
        cfg = ServerConfig(
            num_cores=num_cores,
            num_workers=num_workers,
            integration=integration,
            endpoints=[
                EndpointConfig(
                    task=task,
                    name=f"{task}",
                    model=model_path,
                    batch_size=batch_size,
                )
            ],
            loggers={},
        )

        # saving yaml config to temporary directory
        with TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "server-config.yaml")
            with open(config_path, "w") as fp:
                yaml.dump(cfg.model_dump(), fp)

            server = _fetch_server(integration=integration, config=config_path)
            server.start_server(
                host, port, log_level, hot_reload_config=hot_reload_config
            )
    else:
        raise ValueError("Must specify either --task or --config_file. Found neither")


def _fetch_server(integration: str, config: Union[ServerConfig, str]):
    if isinstance(config, str):
        with open(config) as fp:
            obj = yaml.safe_load(fp)

        config = ServerConfig(**obj)

    if config.integration:
        # if the cli argument provided is not local, use the cli argument
        # otherwise, override with the value in the config file. This gives the
        # cli precedence.
        if integration == INTEGRATION_LOCAL:
            integration = config.integration

    if integration == INTEGRATION_LOCAL:
        return DeepsparseServer(server_config=config)
    elif integration == INTEGRATION_SAGEMAKER:
        return SagemakerServer(server_config=config)
    elif integration == INTEGRATION_OPENAI:
        return OpenAIServer(server_config=config)
    else:
        raise ValueError(
            f"{integration} is not a supported integration. Must be "
            f"one of {INTEGRATIONS}."
        )


if __name__ == "__main__":
    main()
