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

import importlib
import logging
from typing import Any, Dict, List, Optional, Type

from deepsparse.loggers import (
    AsyncLogger,
    BaseLogger,
    FunctionLogger,
    MultiLogger,
    PrometheusLogger,
    PythonLogger,
)
from deepsparse.loggers.helpers import get_function_and_function_name
from deepsparse.server.config import (
    MetricFunctionConfig,
    ServerConfig,
    SystemLoggingConfig,
    SystemLoggingGroup,
)


__all__ = ["build_logger", "custom_logger_from_identifier", "default_logger"]

_LOGGER = logging.getLogger(__name__)
_LOGGER_MAPPING = {"python": PythonLogger, "prometheus": PrometheusLogger}


def custom_logger_from_identifier(custom_logger_identifier: str) -> Type[BaseLogger]:
    """
    Parse the custom logger identifier in order to import a custom logger class object
    from the user-specified python script

    :param custom_logger_identifier: string in the form of
           '<path_to_the_python_script>:<custom_logger_class_name>
    :return: custom logger class object
    """
    path, logger_object_name = custom_logger_identifier.split(":")
    spec = importlib.util.spec_from_file_location("user_defined_custom_logger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, logger_object_name)


def default_logger() -> Dict[str, BaseLogger]:
    """
    :return: default PythonLogger object for the deployment scenario
    """
    _LOGGER.info("Created default logger: PythonLogger")
    return {"python": PythonLogger()}


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
    system_logging_config = server_config.system_logging

    leaf_loggers = (
        build_leaf_loggers(loggers_config) if loggers_config else default_logger()
    )

    function_loggers_data = build_data_loggers(server_config.endpoints, leaf_loggers)
    function_loggers_system = build_system_loggers(leaf_loggers, system_logging_config)
    function_loggers = function_loggers_data + function_loggers_system

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


def build_data_loggers(
    endpoints: List["EndpointConfig"], loggers: Dict[str, BaseLogger]  # noqa F821
) -> List[FunctionLogger]:
    """
    Build a set of data loggers (FunctionLogger instances)
    according to the configuration.

    :param endpoints: A list of server's endpoint configurations;
        the configurations contain the information about the metric
        functions (MetricFunctionConfig objects)
        and the targets that the functions are to be applied to
    :param loggers: The created "leaf" loggers
    :return: A list of FunctionLogger instances responsible
        for logging data information
    """
    data_loggers = []
    for endpoint in endpoints:
        if endpoint.data_logging is None:
            continue
        for target, metric_functions in endpoint.data_logging.items():
            target_identifier = _get_target_identifier(target, endpoint.name)
            for metric_function in metric_functions:
                data_loggers.append(
                    _build_function_logger(metric_function, target_identifier, loggers)
                )
    return data_loggers


def build_system_loggers(
    loggers: Dict[str, BaseLogger], system_logging_config: SystemLoggingConfig
) -> List[FunctionLogger]:
    """
    Create a system loggers according to the configuration specified
    in `system_logging_config`. System loggers are FunctionLogger instances
    responsible for logging system groups metrics.

    :param loggers: The created "leaf" loggers
    :param system_logging_config: The system logging configuration
    :return: A list of FunctionLogger instances responsible for logging system data
    """
    system_loggers = []
    system_logging_group_names = []
    if not system_logging_config.enable:
        return system_loggers

    for config_group_name, config_group_args in system_logging_config:
        if not isinstance(config_group_args, SystemLoggingGroup):
            continue
        if not config_group_args.enable:
            continue

        system_loggers.append(
            _build_function_logger(
                metric_function_cfg=MetricFunctionConfig(
                    func="identity",
                    frequency=1,
                    target_loggers=config_group_args.target_loggers,
                ),
                target_identifier=config_group_name,
                loggers=loggers,
            )
        )
        system_logging_group_names.append(config_group_name)

    _LOGGER.info("System Logging: enabled for groups: %s", system_logging_group_names)

    return system_loggers


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


def _get_target_identifier(
    target_name: str, endpoint_name: Optional[str] = None
) -> str:
    if target_name.startswith("re:"):
        # if target name starts with "re:", it is a regex,
        # and we don't need to add the endpoint name to it
        return target_name
    if endpoint_name:
        return f"{endpoint_name}/{target_name}"
    return target_name
