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
There are two sub-commands for the server:
1. `deepsparse.server config [OPTIONS] <config path>`
2. `deepsparse.server task [OPTIONS] <task>
```
"""

import os
import warnings
from tempfile import TemporaryDirectory
from typing import Optional

import click
import yaml

from deepsparse.pipeline import SupportedTasks
from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.server import start_server


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
    type=click.Choice(["local", "sagemaker"], case_sensitive=False),
    default="local",
    help=(
        "Name of deployment integration that this server will be deployed to "
        "Currently supported options are 'default' and 'sagemaker' for "
        "inference deployment with Amazon Sagemaker"
    ),
)


@click.group(
    invoke_without_command=True,
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"), show_default=True
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
         text_log_save_freq: 30
     endpoints:
     - task: question_answering
       ...
    ```
    """
    if ctx.invoked_subcommand is not None:
        return

    if task is None and config_file is None:
        raise ValueError("Must specify either --task or --config_file. Found neither")

    if task is not None:
        cfg = ServerConfig(
            num_cores=num_cores,
            num_workers=num_workers,
            integration=integration,
            endpoints=[
                EndpointConfig(
                    task=task,
                    name=f"{task}",
                    route="/predict",
                    model=model_path,
                    batch_size=batch_size,
                )
            ],
            loggers={},
        )

        with TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "server-config.yaml")
            with open(config_path, "w") as fp:
                yaml.dump(cfg.dict(), fp)
            start_server(
                config_path, host, port, log_level, hot_reload_config=hot_reload_config
            )

    if config_file is not None:
        start_server(
            config_file, host, port, log_level, hot_reload_config=hot_reload_config
        )


@main.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"), show_default=True
    ),
)
@click.argument("config-path", type=str)
@HOST_OPTION
@PORT_OPTION
@LOG_LEVEL_OPTION
@HOT_RELOAD_OPTION
def config(
    config_path: str, host: str, port: int, log_level: str, hot_reload_config: bool
):
    "[DEPRECATED] Run the server using configuration from a .yaml file."
    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn(
        "Using the `config` sub command is deprecated. "
        "Use the `--config_file` argument instead.",
        category=DeprecationWarning,
    )
    start_server(
        config_path, host, port, log_level, hot_reload_config=hot_reload_config
    )


@main.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"), show_default=True
    ),
)
@click.argument(
    "task",
    type=click.Choice(SupportedTasks.task_names(), case_sensitive=False),
)
@MODEL_OPTION
@BATCH_OPTION
@CORES_OPTION
@WORKERS_OPTION
@HOST_OPTION
@PORT_OPTION
@LOG_LEVEL_OPTION
@HOT_RELOAD_OPTION
@INTEGRATION_OPTION
def task(
    task: str,
    model_path: str,
    batch_size: int,
    num_cores: int,
    num_workers: int,
    host: str,
    port: int,
    log_level: str,
    hot_reload_config: bool,
    integration: str,
):
    """
    [DEPRECATED] Run the server using configuration with CLI options,
    which can only serve a single model.
    """

    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn(
        "Using the `task` sub command is deprecated. "
        "Use the `--task` argument instead.",
        category=DeprecationWarning,
    )

    cfg = ServerConfig(
        num_cores=num_cores,
        num_workers=num_workers,
        integration=integration,
        endpoints=[
            EndpointConfig(
                task=task,
                name=f"{task}",
                route="/predict",
                model=model_path,
                batch_size=batch_size,
            )
        ],
        loggers={},
    )

    with TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "server-config.yaml")
        with open(config_path, "w") as fp:
            yaml.dump(cfg.dict(), fp)
        start_server(
            config_path, host, port, log_level, hot_reload_config=hot_reload_config
        )


if __name__ == "__main__":
    main()
