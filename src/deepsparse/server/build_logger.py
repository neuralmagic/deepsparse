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

from typing import Any, Dict, List, Union

from deepsparse import loggers as logger_objects
from deepsparse.loggers.config import MetricFunctionConfig
from deepsparse.server.config import ServerConfig


__all__ = ["build_logger"]

_LOGGER_MAPPING = {"python": logger_objects.PythonLogger}


def build_logger(server_config: ServerConfig) -> logger_objects.BaseLogger:
    """
    Builds a DeepSparse logger from the ServerConfig.

    The process follows the tree-like hierarchy:

    First: the "leaf" loggers are being built.
    Second: if specified in ServerConfig (inside the config
        of every server endpoint; under 'data_logging' value),
        every "leaf" logger is wrapped inside a FunctionLogger.
    Third: if required, the list of resulting loggers is wrapped inside a
        MultiLogger.
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
        target_to_metric_function_configs = _get_target_to_metric_function_configs(
            endpoints=server_config.endpoints, logger_name=name
        )
        if target_to_metric_function_configs:
            logger = logger_objects.FunctionLogger(
                logger=leaf_logger,
                target_to_metric_function_configs=target_to_metric_function_configs,
            )
        else:
            logger = leaf_logger

        loggers.append(logger)

    return logger_objects.MultiLogger(loggers) if len(loggers) > 1 else loggers[0]


def build_leaf_loggers(
    loggers_config: List[Union[str, Dict[str, Dict[str, Any]]]],
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
            leaf_logger = _LOGGER_MAPPING[logger_name]()
        else:
            # instantiating a logger with arguments
            logger_name, logger_arguments = tuple(logger_config.items())[0]
            leaf_logger = _LOGGER_MAPPING[logger_name](**logger_arguments)
        loggers.update({logger_name: leaf_logger})
    return loggers


def _get_target_to_metric_function_configs(
    endpoints: List["EndpointConfig"], logger_name: str  # noqa F821
):
    target_to_metric_function_configs = {}
    for endpoint in endpoints:
        if endpoint.data_logging is None:
            continue
        for target_logging_cfg in endpoint.data_logging:
            target_identifier = _get_target_identifier(
                endpoint_name=endpoint.name, target_name=target_logging_cfg.target
            )
            functions = [
                MetricFunctionConfig.from_server(**function)
                for function in target_logging_cfg.functions
            ]
            # remove those functions that have their own logger configuration,
            # and it does not include `logger_name`
            functions = [
                function
                for function in functions
                if function.loggers is not None and logger_name in function.loggers
            ]
            target_to_metric_function_configs[target_identifier] = functions
    return target_to_metric_function_configs


def _get_target_identifier(endpoint_name: str, target_name: str) -> str:
    return f"{endpoint_name}.{target_name}"
