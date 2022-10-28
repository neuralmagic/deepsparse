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
Specifies the mapping from the ServerConfig to the DeepSparse Logger
"""

from typing import Any, Dict, Optional, Union

from deepsparse import loggers as logger_objects
from deepsparse.loggers.config import (
    MultiplePipelinesLoggingConfig,
    PipelineLoggingConfig,
)
from deepsparse.server.config import ServerConfig


__all__ = ["build_logger"]

SUPPORTED_LOGGER_NAMES = ["python"]


def build_logger(server_config: ServerConfig) -> logger_objects.BaseLogger:
    """
    Builds a DeepSparse logger from the ServerConfig.

    The process follows the following tree-like hierarchy:

    First: the "leaf" loggers are being built.
    Second: if there are multiple "leaf" loggers,
            they are wrapped inside the MultiLogger. (TODO)
            else: we continue with the single "leaf" logger.
    Third: if specified in the ServerConfig, the resulting logger is wrapped inside
            the FunctionLogger.
    Fourth: lastly, the resulting loggers is wrapped inside the SubprocessLogger (TODO)

    :param server_config: the Server configuration model
    :return: a DeepSparse logger instance
    """

    loggers_config = server_config.loggers
    if loggers_config is None:
        return loggers_config

    multi_pipelines_logging = get_multiple_pipelines_logging_config(server_config)

    loggers = []
    for logger_config in loggers_config:
        if isinstance(logger_config, str):
            # instantiating a logger without arguments
            logger_name = logger_config
            leaf_logger = _build_single_logger(logger_name=logger_name)
        else:
            # instantiating a logger with arguments
            logger_name, logger_arguments = tuple(logger_config.items())[0]
            leaf_logger = _build_single_logger(
                logger_name=logger_name, logger_arguments=logger_arguments
            )

        loggers.append(leaf_logger)

    # logger = logger_objects.MultiLogger(loggers) if len(loggers) > 1 else loggers[0]
    logger = loggers[0]
    logger = (
        logger_objects.FunctionLogger(logger=logger, config=multi_pipelines_logging)
        if multi_pipelines_logging
        else logger
    )

    return logger


def get_multiple_pipelines_logging_config(
    server_config: ServerConfig,
) -> MultiplePipelinesLoggingConfig:
    """
    Create a MultiplePipelinesLoggingConfig from the ServerConfig

    :param server_config: the Server configuration model
    :return: a MultiplePipelinesLoggingConfig model
    """
    pipeline_logging_configs = []
    endpoints = server_config.endpoints
    for endpoint in endpoints:
        name = endpoint.name or endpoint.task
        targets = endpoint.data_logging
        if not targets:
            continue

        pipeline_logging_configs.append(
            PipelineLoggingConfig(name=name, targets=targets)
        )
    if not pipeline_logging_configs:
        return None

    return MultiplePipelinesLoggingConfig(pipelines=pipeline_logging_configs)


def _build_single_logger(
    logger_name: str, logger_arguments: Optional[Dict[str, Any]] = None
) -> Union[logger_objects.PythonLogger]:
    if logger_name == "python":
        if logger_arguments:
            raise ValueError(
                "Attempting to create PythonLogger. "
                "The logger takes no arguments. "
                f"Attempting to pass arguments: {logger_arguments}"
            )
        return logger_objects.PythonLogger()

        raise ValueError(
            "Attempting to create a DeepSparse Logger with "
            f"an unknown logger_name: {logger_name}. "
            f"Supported names are: {SUPPORTED_LOGGER_NAMES}"
        )
