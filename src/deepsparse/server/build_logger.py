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

from typing import Any, Dict, List, Optional, Union

from deepsparse import loggers as logger_objects
from deepsparse.loggers.config import MetricFunctionConfig, TargetLoggingConfig
from deepsparse.server.config import ServerConfig


__all__ = ["build_logger"]

SUPPORTED_LOGGER_NAMES = ["python"]


def build_logger(server_config: ServerConfig) -> logger_objects.BaseLogger:
    """
    Builds a DeepSparse logger from the ServerConfig.

    The process follows the following tree-like hierarchy:

    First: the "leaf" loggers are being built.
    Second: if specified in ServerConfig (key "data_logging"),
        every "leaf" logger is wrapped inside a FunctionLogger
    Third: if required, the list of resulting loggers is wrapped inside a
        MultiLogger
    Fourth: if specified, the resulting logger is wrapped inside the SubprocessLogger

    :param server_config: the Server configuration model
    :return: a DeepSparse logger instance
    """

    loggers_config = server_config.loggers
    if loggers_config is None:
        return loggers_config

    leaf_loggers = build_leaf_loggers(loggers_config)
    loggers = []
    for name, leaf_logger in leaf_loggers.items():
        target_logging_configs = _get_target_logging_configs(name, server_config)

        if target_logging_configs:
            logger = logger_objects.FunctionLogger(
                logger=leaf_logger, target_logging_configs=target_logging_configs
            )
        else:
            logger = leaf_logger

        loggers.append(logger)
    return logger_objects.MultiLogger(loggers) if len(loggers) > 1 else loggers[0]


def build_leaf_loggers(
    loggers_config: List[Union[str, Dict[str, Dict[str, Any]]]]
) -> Dict[str, logger_objects.BaseLogger]:
    """
    Instantiate a set of leaf loggers according to the configuration

    :param loggers_config: Data structure, that specifies the leaf loggers to
        be instantiated
    :return: A dictionary that contains a mapping from a logger's name to its instance
    """

    loggers = {}
    for logger_config in loggers_config:
        if isinstance(logger_config, str):
            # instantiating a logger without arguments
            logger_name = logger_config
            leaf_logger = _build_single_logger(logger_name=logger_name)
        else:
            # instantiating a logger with arguments
            logger_name, logger_arguments = tuple(logger_config.items())[
                0
            ]  # check if could be more elegant
            leaf_logger = _build_single_logger(
                logger_name=logger_name, logger_arguments=logger_arguments
            )
        loggers.update(leaf_logger)
    return loggers


def _get_target_logging_configs(
    leaf_logger_name: str, server_config: ServerConfig
) -> List[TargetLoggingConfig]:
    list_target_logging_configs = []
    # iterate over all endpoints
    for endpoint in server_config.endpoints:
        if endpoint.data_logging is None:
            continue
        # if endpoint has data logging information,
        # iterate over endpoint's target logging config
        for target_logging_config in endpoint.data_logging:
            # new target name is a composition of the endpoint name and target name
            target_identifier = f"{endpoint.name}.{target_logging_config.target}"
            # get metric logging config corresponding to the leaf logger
            metric_function_configs = _get_metric_function_configs(
                leaf_logger_name, target_logging_config.mappings
            )
            # append to the result
            list_target_logging_configs.append(
                TargetLoggingConfig(
                    target=target_identifier, mappings=metric_function_configs
                )
            )
    return list_target_logging_configs


def _get_metric_function_configs(
    leaf_logger_name: str, metric_function_configs: List[MetricFunctionConfig]
) -> List[MetricFunctionConfig]:
    list_metric_logging_configs = []
    # iterate over all metric function configs
    for metric_logging_config in metric_function_configs:
        if metric_logging_config.logger is not None:
            # check whether a metric function is specified to log
            # only to a set of specific leaf loggers
            if leaf_logger_name not in metric_logging_config.logger:
                continue

        list_metric_logging_configs.append(metric_logging_config)
    return list_metric_logging_configs


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
        return {logger_name: logger_objects.PythonLogger()}

    raise ValueError(
        "Attempting to create a DeepSparse Logger with "
        f"an unknown logger_name: {logger_name}. "
        f"Supported names are: {SUPPORTED_LOGGER_NAMES}"
    )
