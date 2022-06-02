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
Configurations for serving models in the DeepSparse inference server
"""

import json
import os
from functools import lru_cache
from typing import List

import yaml
from pydantic import BaseModel, Field

from deepsparse import PipelineConfig
from deepsparse.cpu import cpu_architecture


__all__ = [
    "ENV_DEEPSPARSE_SERVER_CONFIG",
    "ENV_SINGLE_PREFIX",
    "ServerConfig",
]


ENV_DEEPSPARSE_SERVER_CONFIG = "DEEPSPARSE_SERVER_CONFIG"
ENV_SINGLE_PREFIX = "DEEPSPARSE_SINGLE_MODEL:"


class ServerConfig(BaseModel):
    """
    A configuration for serving models in the DeepSparse inference server
    """

    models: List[PipelineConfig] = Field(
        default=[],
        description=(
            "The models to serve in the server defined by PipelineConfig objects"
        ),
    )
    workers: str = Field(
        default=max(1, cpu_architecture().num_available_physical_cores // 2),
        description=(
            "The number of maximum workers to use for processing pipeline requests. "
            "Defaults to the number of physical cores on the device."
        ),
    )
    integration: str = Field(
        default="default",
        description=(
            "Name of deployment integration that this server will be deployed to "
            "Currently supported options are None for default inference and "
            "'sagemaker' for inference deployment with AWS Sagemaker"
        ),
    )


@lru_cache()
def server_config_from_env(env_key: str = ENV_DEEPSPARSE_SERVER_CONFIG):
    """
    Load a server configuration from the targeted environment variable given by env_key.

    :param env_key: the environment variable to load the configuration from.
        Defaults to ENV_DEEPSPARSE_SERVER_CONFIG
    :return: the loaded configuration file
    """
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
            PipelineConfig(
                task=config_dict["task"],
                model_path=config_dict["model_path"],
                batch_size=config_dict["batch_size"],
            )
        )
    else:
        with open(config_file) as file:
            config_dict = yaml.safe_load(file.read())
        config_dict["models"] = (
            [PipelineConfig(**model) for model in config_dict["models"]]
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
    integration: str,
    env_key: str = ENV_DEEPSPARSE_SERVER_CONFIG,
):
    """
    Put a server configuration in an environment variable given by env_key.
    If config_file is given, ignores task, model_path, and batch_size.
    Otherwise, creates a configuration file from task, model_path, and batch_size
    for serving a single model.

    :param config_file: the path to the config file to store in the environment
    :param task: the task the model_path is serving such as question_answering.
        If config_file is supplied, this is ignored.
    :param model_path: the path to a model.onnx file, a model folder containing
        the model.onnx and supporting files, or a SparseZoo model stub.
        If config_file is supplied, this is ignored.
    :param batch_size: the batch size to serve the model from model_path with.
        If config_file is supplied, this is ignored.
    :param integration: name of deployment integration that this server will be
        deployed to. Supported options include None for default inference and
        sagemaker for inference deployment on AWS Sagemaker
    :param env_key: the environment variable to set the configuration in.
        Defaults to ENV_DEEPSPARSE_SERVER_CONFIG
    """
    if config_file is not None:
        config = config_file
    else:
        if task is None or model_path is None:
            raise ValueError(
                "config_file not given, model_path and task both must be supplied "
                "for serving"
            )

        single_str = json.dumps(
            {
                "task": task,
                "model_path": model_path,
                "batch_size": batch_size,
                "integration": integration,
            }
        )
        config = f"{ENV_SINGLE_PREFIX}{single_str}"

    os.environ[env_key] = config
