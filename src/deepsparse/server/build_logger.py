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

import logging
from typing import Any, Dict, List

from deepsparse import BaseLogger, FunctionLogger, MultiLogger, PythonLogger
from deepsparse.loggers.helpers import get_function_and_function_name
from deepsparse.server.config import ServerConfig


__all__ = ["build_logger"]

_LOGGER = logging.getLogger(__name__)

_LOGGER_MAPPING = {"python": PythonLogger}


def build_logger(server_config: ServerConfig) -> BaseLogger:
    """
    Builds a DeepSparse logger from the ServerConfig.

    The process follows the following hierarchy:

    First: if global logger config is provided,
        the "leaf" loggers are built.

    Second: if data logging config is specified, a set of
        function loggers wraps around the appropriate "leaf" loggers.

    Third: if required, the list of resulting loggers is wrapped inside a
        MultiLogger.

    :param server_config: the Server configuration model
    :return: a DeepSparse logger instance
    """

    loggers_config = server_config.loggers
    if not loggers_config:
        return None
    leaf_loggers = build_leaf_loggers(loggers_config)

    loggers = build_function_loggers(server_config.endpoints, leaf_loggers)
    if not loggers:
        loggers = list(leaf_loggers.values())

    _LOGGER.info("Created logger from the config")
    return MultiLogger(loggers) if len(loggers) > 1 else loggers[0]


def build_leaf_loggers(
    loggers_config: Dict[str, Dict[str, Any]]
) -> Dict[str, BaseLogger]:
    """
    Instantiate a set of leaf loggers according to the configuration

    :param loggers_config: Config; specifies the leaf loggers to be instantiated
    :return: A dictionary that contains a mapping from a logger's name to its instance
    """
    loggers = {}
    for logger_name, logger_arguments in loggers_config.items():
        leaf_logger = _LOGGER_MAPPING[logger_name](**logger_arguments)
        loggers.update({logger_name: leaf_logger})
    return loggers


def build_function_loggers(
    endpoints: List["EndpointConfig"], loggers: Dict[str, BaseLogger]  # noqa F821
) -> List[FunctionLogger]:
    """
    Build a set of function loggers according to the configuration.

    :param endpoints: A list of server's endpoint configurations;
        the configurations contain the information about the metric
        functions and the targets that the functions are to be applied to
    :param loggers: The created "leaf" loggers
    :return: A list of FunctionLogger instances
    """
    function_loggers = []
    for endpoint in endpoints:
        if endpoint.data_logging is None:
            continue
        for target_logging_cfg in endpoint.data_logging:
            target = target_logging_cfg.target
            target_identifier = _get_target_identifier(endpoint.name, target)
            function_cfgs = target_logging_cfg.functions
            for function_cfg in function_cfgs:
                function_loggers.append(
                    _build_function_logger(function_cfg, target_identifier, loggers)
                )
    return function_loggers


def _build_function_logger(
    function_cfg: Dict[str, Any], target_identifier: str, loggers=Dict[str, BaseLogger]
) -> FunctionLogger:
    # automatically extract function and function name
    # from the function_identifier
    function, function_name = get_function_and_function_name(function_cfg["func"])
    # if metric function has attribute `target_loggers`,
    # override the global logger configuration
    if function_cfg.get("target_loggers"):
        target_loggers = [
            leaf_logger
            for name, leaf_logger in loggers.items()
            if name in function_cfg["target_loggers"]
        ]
    else:
        target_loggers = list(loggers.values())

    # if `target loggers` is a list len > 1, wrap it inside a MultiLogger
    target_logger = (
        MultiLogger(loggers=target_loggers)
        if len(target_loggers) > 1
        else target_loggers[0]
    )
    return FunctionLogger(
        logger=target_logger,
        target_identifier=target_identifier,
        function=function,
        function_name=function_name,
        frequency=function_cfg["frequency"],
    )


def _get_target_identifier(endpoint_name: str, target_name: str) -> str:
    return f"{endpoint_name}.{target_name}"
