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

from typing import Any, Dict, List, Optional

from deepsparse.loggers import (
    AsyncLogger,
    BaseLogger,
    FunctionLogger,
    MetricCategories,
    MultiLogger,
    PrometheusLogger,
    PythonLogger,
)
from deepsparse.loggers.helpers import get_function_and_function_name
from deepsparse.server.config import MetricFunctionConfig, ServerConfig
from deepsparse.server.helpers import custom_logger_from_identifier, default_logger


__all__ = ["build_logger"]

_LOGGER_MAPPING = {"python": PythonLogger, "prometheus": PrometheusLogger}


def build_logger(server_config: ServerConfig) -> BaseLogger:
    """
    Builds a DeepSparse logger from the ServerConfig.

    The process follows the following hierarchy:

    First: if global logger config is provided,
        the "leaf" loggers are built.

    Second: if data logging config is specified, a set of
        function loggers wraps around the appropriate "leaf" loggers.

    Third: The resulting loggers are wrapped inside a MultiLogger.

    :param server_config: the Server configuration model
    :return: a DeepSparse logger instance
    """

    loggers_config = server_config.loggers
    if not loggers_config:
        return AsyncLogger(
            logger=MultiLogger(
                [default_logger()]
            ),  # wrap all loggers to async log call
            max_workers=1,
        )

    # base level loggers that log raw values for monitoring. ie python, prometheus
    leaf_loggers = build_leaf_loggers(loggers_config)

    function_loggers = build_function_loggers(server_config.endpoints, leaf_loggers)

    # add logger to ensure leaf level logging of all system (timing) logs
    function_loggers.append(_create_system_logger(leaf_loggers))

    return AsyncLogger(
        logger=MultiLogger(function_loggers),  # wrap all loggers to async log call
        max_workers=1,
    )


def build_leaf_loggers(
    loggers_config: Dict[str, Optional[Dict[str, Any]]]
) -> Dict[str, BaseLogger]:
    """
    Instantiate a set of leaf loggers according to the configuration

    :param loggers_config: Config; specifies the leaf loggers to be instantiated
    :return: A dictionary that contains a mapping from a logger's name to its instance
    """
    loggers = {}
    for logger_name, logger_arguments in loggers_config.items():
        path_custom_logger = (
            logger_arguments.get("path")
            if logger_arguments is not None
            else logger_arguments
        )
        if path_custom_logger:
            # if `path` argument is provided, use the custom logger
            leaf_logger = _build_custom_logger(logger_arguments)
        else:
            # otherwise, use the built-in logger
            logger_to_instantiate = _LOGGER_MAPPING.get(logger_name)
            if logger_to_instantiate is None:
                raise ValueError(
                    f"Unknown logger name: {logger_name}. "
                    f"supported logger names: {list(_LOGGER_MAPPING.keys())}"
                )
            logger_arguments = {} if logger_arguments is None else logger_arguments
            leaf_logger = logger_to_instantiate(**logger_arguments)
        loggers.update({logger_name: leaf_logger})
    return loggers


def build_function_loggers(
    endpoints: List["EndpointConfig"], loggers: Dict[str, BaseLogger]  # noqa F821
) -> List[FunctionLogger]:
    """
    Build a set of function loggers according to the configuration.

    :param endpoints: A list of server's endpoint configurations;
        the configurations contain the information about the metric
        functions (MetricFunctionConfig objects)
        and the targets that the functions are to be applied to
    :param loggers: The created "leaf" loggers
    :return: A list of FunctionLogger instances
    """
    function_loggers = []
    for endpoint in endpoints:
        if endpoint.data_logging is None:
            continue
        for target, metric_functions in endpoint.data_logging.items():
            target_identifier = _get_target_identifier(endpoint.name, target)
            for metric_function in metric_functions:
                function_loggers.append(
                    _build_function_logger(metric_function, target_identifier, loggers)
                )
    return function_loggers


def _create_system_logger(loggers: Dict[str, BaseLogger]) -> FunctionLogger:
    # returns a function logger that matches to all system logs, logging
    # every system call to each leaf logger
    return _build_function_logger(
        metric_function_cfg=MetricFunctionConfig(
            func="identity", frequency=1, target_loggers=None
        ),
        target_identifier=f"category:{MetricCategories.SYSTEM.value}",
        loggers=loggers,
    )


def _build_function_logger(
    metric_function_cfg: MetricFunctionConfig,
    target_identifier: str,
    loggers: Dict[str, BaseLogger],
) -> FunctionLogger:
    # automatically extract function and function name
    # from the function_identifier
    function, function_name = get_function_and_function_name(metric_function_cfg.func)
    # if metric function has attribute `target_loggers`,
    # override the global logger configuration
    if metric_function_cfg.target_loggers:
        target_loggers = [
            leaf_logger
            for name, leaf_logger in loggers.items()
            if name in metric_function_cfg.target_loggers
        ]
    else:
        target_loggers = list(loggers.values())

    return FunctionLogger(
        logger=MultiLogger(loggers=target_loggers),
        target_identifier=target_identifier,
        function=function,
        function_name=function_name,
        frequency=metric_function_cfg.frequency,
    )


def _build_custom_logger(logger_arguments: Dict[str, Any]) -> BaseLogger:
    # gets the identifier from logger arguments and simultaneously
    # removes the identifier from the arguments
    custom_logger_identifier = logger_arguments.pop("path")
    logger = custom_logger_from_identifier(custom_logger_identifier)(**logger_arguments)
    if not isinstance(logger, BaseLogger):
        raise ValueError(
            f"Custom logger must be a subclass of BaseLogger. "
            f"Got {type(logger)} instead."
        )
    return logger


def _get_target_identifier(endpoint_name: str, target_name: str) -> str:
    if target_name.startswith("re:"):
        # if target name starts with "re:", it is a regex,
        # and we don't need to add the endpoint name to it
        return target_name
    return f"{endpoint_name}/{target_name}"
