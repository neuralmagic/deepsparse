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

import json
import os
from functools import lru_cache
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field

from deepsparse.cpu import cpu_architecture


__all__ = [
    "ENV_DEEPSPARSE_SERVER_CONFIG",
    "ENV_SINGLE_PREFIX",
    "ServeModelConfig",
    "ServerConfig",
]


ENV_DEEPSPARSE_SERVER_CONFIG = "DEEPSPARSE_SERVER_CONFIG"
ENV_SINGLE_PREFIX = "DEEPSPARSE_SINGLE_MODEL:"


class ServeModelConfig(BaseModel):
    task: str = Field(
        description=(
            "The task the model_path is serving. For example, one of: "
            "question_answering, text_classification, token_classification."
        ),
    )
    model_path: str = Field(
        description=(
            "The path to a model.onnx file, "
            "a model folder containing the model.onnx and supporting files, "
            "or a SparseZoo model stub."
        ),
    )
    batch_size: int = Field(
        default=1,
        description=(
            "The batch size to instantiate the model with and use for serving"
        ),
    )
    alias: str = Field(
        default=None,
        description=(
            "Alias name for model pipeline to be served. A convenience route of "
            "/predict/alias will be added to the server if present. "
        ),
    )
    kwargs: Dict[str, Any] = Field(
        default={},
        description=(
            "Additional arguments for inference with the model that will be passed "
            "into the pipeline as kwargs"
        ),
    )
    engine: str = Field(
        default="deepsparse",
        description=(
            "The engine to use for serving the models such as deepsparse or onnxruntime"
        ),
    )
    num_cores: int = Field(
        default=None,
        description=(
            "The number of physical cores to restrict the DeepSparse Engine to. "
            "Defaults to all cores."
        ),
    )
    scheduler: str = Field(
        default="async",
        description=(
            "The scheduler to use with the DeepSparse Engine such as sync or async. "
            "Defaults to async"
        ),
    )


class ServerConfig(BaseModel):
    models: List[ServeModelConfig] = Field(
        default=[],
        description=(
            "The models to serve in the server defined by the additional arguments"
        ),
    )
    workers: str = Field(
        default=max(1, cpu_architecture().num_available_physical_cores // 2),
        description=(
            "The number of maximum workers to use for processing pipeline requests. "
            "Defaults to the number of physical cores on the device."
        ),
    )


@lru_cache()
def server_config_from_env(env_key: str = ENV_DEEPSPARSE_SERVER_CONFIG):
    config_file = os.environ[env_key]

    if not config_file:
        raise ValueError(
            "environment variable for deepsparse server config not found at "
            f"{env_key}"
        )

    if config_file.startswith(ENV_SINGLE_PREFIX):
        config_dict = json.loads(config_file.replace(ENV_SINGLE_PREFIX, ""))
        config = ServerConfig()
        config.models.append(
            ServeModelConfig(
                task=config_dict["task"],
                model_path=config_dict["model_path"],
                batch_size=config_dict["batch_size"],
            )
        )
    else:
        config_dict = yaml.safe_load(config_file)
        config_dict["models"] = (
            [ServeModelConfig(**model) for model in config_dict["models"]]
            if "models" in config_dict
            else []
        )
        config = ServerConfig(**config_dict)

    if len(config.models) == 0:
        raise ValueError(
            "There must be at least one model to serve in the configuration "
            "for the deepsparse inference server"
        )

    return config


def server_config_to_env(
    config_file: str,
    task: str,
    model_path: str,
    batch_size: int,
    env_key: str = ENV_DEEPSPARSE_SERVER_CONFIG,
):
    if config_file is not None:
        config = config_file
    else:
        if task is None or model_path is None:
            raise ValueError(
                "config_file not given, model_path and task both must be supplied "
                "for serving"
            )

        single_str = json.dumps(
            {"task": task, "model_path": model_path, "batch_size": batch_size}
        )
        config = f"{ENV_SINGLE_PREFIX}{single_str}"

    os.environ[env_key] = config
