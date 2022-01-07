"""
File containing utility classes and functions for DeepSparse Inference Server
"""

import argparse
from functools import lru_cache
from typing import TypeVar, Optional

from deepsparse.transformers import Pipeline, pipeline
import numpy as np
from pydantic import BaseModel, BaseSettings, Field

__all__ = [
    'parse_api_settings',
    'PipelineConfig',
    'fix_numpy_types'
]

################################################################################
### APIConfig
################################################################################

WORKERS = 1


class APIConfig(BaseModel):
    """
    API level settings for server

    :param host: str representing the host URL for server deployment. Defaults
        to `0.0.0.0`
    :param port: int representing the port number to use for server deployment.
        Defaults to port 5543. Note-> the port must be available for use
    :param workers: int The number of server processes to spawn. Is set to a
        constant value of 1 to ensure only one copy of backend deepsparse engine
        is running at all times.
    """
    host: str = '0.0.0.0'
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


################################################################################
### PipelineConfig
################################################################################

DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_LENGTH = 128
PipelineConfigType = TypeVar(
    'PipelineEngineSettingsType',
    bound='PipelineEngineSettings',
)


class PipelineConfig(BaseSettings):
    """
    Settings for pipeline should be set via .env file
    using the following steps:
    ```
    set -a
    export <file>.env
    set +a
    ```

    :param model_file_or_stub: path to (ONNX) model file or SparseZoo stub
    :param task: name of the task to define which pipeline to create. Defaults
        to "question-answering"
    :param num_cores: number of CPU cores to run engine with. Default is the
        maximum available
    :param batch_size: The batch size to use for pipeline. Defaults to 1.
        Note: "question-answering" only supports batch_size 1
    :param max_length: maximum sequence length of model inputs. Default is 128
    :param scheduler: The scheduler to use for the engine. Can be None, single
        or multi
    :param concurrent_engine_requests: Number of concurrent workers to use
        for sending requests to the Pipeline. Defaults to 3.
    """
    model_file_or_stub: str = 'zoo:nlp/question_answering/bert-base/' \
                              'pytorch/huggingface/squad/' \
                              'pruned_quant_3layers-aggressive_89'
    task: str = 'question-answering'
    num_cores: Optional[int] = None
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    scheduler: str = 'multi'
    concurrent_engine_requests: int = 3

    @staticmethod
    @lru_cache()
    def get_config() -> PipelineConfigType:
        """
        :return: PipelineEngineSettings for setting up the pipeline object
        """
        return PipelineConfig()



################################################################################
### Utility functions
################################################################################

def fix_numpy_types(func):
    """
    Decorator to fix numpy types in Dicts, List[Dicts], List[List[Dicts]]
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
