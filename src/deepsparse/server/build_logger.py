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


import importlib
from collections import defaultdict
from typing import List, Optional

from deepsparse import loggers as logger_objects
from deepsparse.loggers.configs import PipelineLoggingConfig
from deepsparse.server.config import ServerConfig


def build_logger(server_config: ServerConfig) -> logger_objects.BaseLogger:
    """
    Builds a DeepSparse logger from the server config.

    The process follows the following tree-like hierarchy:

    First: the "leaf" loggers are being built
    Second: if there are multiple "leaf" loggers, they are wrapped inside the MultiLogger
            else: we pass on the single "leaf" logger
    Third: if specified in the ServerConfig, the resulting logger is wrapped inside
            the FunctionLogger
    Fourth: lastly, the resulting loggers is wrapped inside the SubprocessLogger (todo)

    :param server_config: the Server configuration model
    :return: a DeepSparse logger instance
    """

    loggers_config = server_config.loggers
    if loggers_config is None:
        return loggers_config

    pipeline_data_logging_configs = get_pipeline_logging_configs(server_config)

    loggers = []
    for logger_config in loggers_config:
        if isinstance(logger_config, str):
            logger_name = logger_config
            loggers.append(_build_single_logger(logger_name=logger_name))
        else:
            logger_name, logger_arguments = tuple(logger_config.items())[0]
            loggers.append(
                _build_single_logger(
                    logger_name=logger_name, logger_arguments=logger_arguments
                )
            )

    logger = logger_objects.MultiLogger(loggers) if len(loggers) > 1 else loggers[0]
    logger = (
        logger_objects.FunctionLogger(
            logger=logger, config=pipeline_data_logging_configs
        )
        if pipeline_data_logging_configs
        else logger
    )

    return logger


def _build_single_logger(logger_name, logger_arguments: Optional = None):
    """
    in future if logger_name == "prometheus":
        return PrometheusLogger(**logger_arguments)
    etc
    """
    if logger_name == "python":
        if logger_arguments:
            raise ValueError()
        return logger_objects.PythonLogger()

    if logger_name == "prometheus":
        return logger_objects.PrometheusLogger(**logger_arguments)
    else:
        raise ValueError()


def get_pipeline_logging_configs(
    server_config: ServerConfig,
) -> List[PipelineLoggingConfig]:
    """
    Create a list of PipelineDataLoggingConfig models (one per each endpoint) from ServerConfig

    :param server_config: a Server configuration model
    :return: a list of PipelineDataLoggingConfig models
    """
    pipeline_data_logging_configs = []
    endpoints = server_config.endpoints
    for endpoint in endpoints:
        pipeline_data_logging_configs.append(
            PipelineLoggingConfig(
                name=endpoint.name or endpoint.task, targets=endpoint.data_logging
            )
        )

    pipeline_names = [config.name for config in pipeline_data_logging_configs]
    from collections import Counter

    counter = Counter(pipeline_names)
    [(k, v) for k, v in counter.items()]

    return pipeline_data_logging_configs
