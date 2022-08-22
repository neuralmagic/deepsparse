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
from typing import Optional

import click
import yaml

from deepsparse.pipeline import Pipeline, SupportedTasks
from deepsparse.server_v2.config import EndpointConfig, ServerConfig
from deepsparse.server_v2.main import start_server


@click.command(
    context_settings=dict(token_normalize_func=lambda x: x.replace("-", "_"))
)
@click.argument(
    "task",
    type=click.Choice(SupportedTasks.task_names(), case_sensitive=False),
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
    "--num-cores",
    type=int,
    default=1,
    help="TODO",
)
@click.option(
    "--num-concurrent-batches",
    type=int,
    default=1,
    help="TODO",
)
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=5543)
@click.option(
    "--log_level",
    type=click.Choice(
        ["debug", "info", "warn", "critical", "fatal"], case_sensitive=False
    ),
    default="info",
    help="Sets the logging level. Defaults to info.",
)
def main(
    task: str,
    model_path: Optional[str],
    batch_size: int,
    num_cores: int,
    num_concurrent_batches: int,
    host: str,
    port: int,
    log_level: str,
):
    cfg = ServerConfig(
        num_cores=num_cores,
        num_concurrent_batches=num_concurrent_batches,
        endpoints=[
            EndpointConfig(
                task=task,
                name=f"{task} inference model",
                endpoint="/predict",
                model=model_path or Pipeline.default_model_for(task),
                batch_size=batch_size,
                accept_multiples_of_batch_size=False,
            )
        ],
    )

    with TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "server-config.yaml")
        with open(config_path, "w") as fp:
            yaml.dump(cfg.dict(), fp)
        start_server(config_path, host, port, log_level)


if __name__ == "__main__":
    main()
