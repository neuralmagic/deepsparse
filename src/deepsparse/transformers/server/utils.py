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
File containing utility classes and functions for DeepSparse Inference Server
"""

import argparse
from functools import lru_cache
from typing import Optional, TypeVar

import numpy as np

from pydantic import BaseModel, BaseSettings, Field


__all__ = ["parse_api_settings", "PipelineEngineConfig", "fix_numpy_types"]

# APIConfig

WORKERS = 1


class APIConfig(BaseModel):
    """
    API level settings for server

    :param host: str representing the host URL for server deployment. Defaults
        to `0.0.0.0`
    :param port: int representing the port number to use for server deployment.
        Defaults to port 5543. Note-> the port must be available for use
    :param workers: int The number of server processes to spawn. Is set to a
        constant value of 1 to ensure only one copy of backend deepsparse ENGINE
        is running at all times.
    """

    host: str = "0.0.0.0"
    port: int = 5543
    workers: int = Field(WORKERS, const=True)


def parse_api_settings() -> APIConfig:
    """
    Adds an ArgumentParser for DeepSparse Inference Server
    and parses APIConfig via Command Line

    :return: APIConfig instance for server config
    """
    parser = argparse.ArgumentParser("Run DeepSparse Inference Server")

    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default="0.0.0.0",
        help="The IP address of the hosted model",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default="5543",
        help="The port that the model is hosted on",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="The number of workers to use for uvicorn",
    )

    _args = parser.parse_args()
    return APIConfig(**vars(_args))


# PipelineEngineConfig

DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_LENGTH = 128
DEFAULT_SCHEDULER = "multi"
PipelineConfigType = TypeVar(
    "PipelineEngineSettingsType",
    bound="PipelineEngineConfig",
)


class PipelineEngineConfig(BaseSettings):
    """
    Settings for pipeline should be set via .env file
    using the following steps:
    ```
    set -a
    export <file>.env
    set +a
    ```

    :param model_file_or_stub: path to (ONNX) model file or SparseZoo stub
    :param task: name of the task to define which pipeline to create.
    :param num_cores: number of CPU cores to run ENGINE with. Default is the
        maximum available
    :param batch_size: The batch size to use for pipeline. Defaults to 1.
        Note: "question-answering" only supports batch_size 1
    :param max_length: maximum sequence length of model inputs. Default is 128
    :param scheduler: The scheduler to use for the ENGINE. Can be None, single
        or multi
    :param concurrent_engine_requests: Number of concurrent workers to use
        for sending requests to the Pipeline. Defaults to 3.
    """

    task: str
    model_file_or_stub: str = (
        "zoo:nlp/question_answering/bert-base/"
        "pytorch/huggingface/squad/"
        "pruned_quant_3layers-aggressive_89"
    )
    num_cores: Optional[int] = None
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    scheduler: str = DEFAULT_SCHEDULER
    concurrent_engine_requests: int = 3

    @staticmethod
    @lru_cache()
    def get_config() -> PipelineConfigType:
        """
        :return: PipelineEngineSettings for setting up the pipeline object
        """
        return PipelineEngineConfig()


# UTILITY FUNCTIONS


def fix_numpy_types(func):
    """
    Decorator to fix numpy types in Dicts, List[Dicts], List[List[Dicts]]
    Because `orjson` does not support serializing individual numpy data types
    yet
    """

    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        def _normalize_fields(_dict):
            if isinstance(_dict, dict):
                for field in _dict:
                    if isinstance(_dict[field], np.generic):
                        _dict[field] = _dict[field].item()

        if isinstance(result, dict):
            _normalize_fields(result)
        elif result and isinstance(result, list):
            for element in result:
                if isinstance(element, list):
                    for _result in element:
                        _normalize_fields(_result)
                else:
                    _normalize_fields(element)

        return result

    return _wrapper
