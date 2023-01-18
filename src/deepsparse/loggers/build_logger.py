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
A general set of functionalities for building complex logger instances to
be used across the repository.
"""

import importlib
import logging
import os
from typing import Any, Dict, List, Optional, Type

import yaml

from deepsparse.loggers import (
    AsyncLogger,
    BaseLogger,
    FunctionLogger,
    MultiLogger,
    PrometheusLogger,
    PythonLogger,
)
from deepsparse.loggers.config import (
    MetricFunctionConfig,
    PipelineLoggingConfig,
    SystemLoggingConfig,
    SystemLoggingGroup,
)
from deepsparse.loggers.helpers import get_function_and_function_name


__all__ = [
    "custom_logger_from_identifier",
    "default_logger",
    "logger_from_config",
    "build_logger",
    "get_target_identifier",
]

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


def logger_from_config(config: str, pipeline_identifier: str = None) -> BaseLogger:
    """
    Builds a pipeline logger from the appropriate configuration file

    :param config: The configuration of the pipeline logger.
        Is a string that represents either:
            a path to the .yaml file
            or
            yaml string representation of the logging config
        The config file should obey the rules enforced by
        the PipelineLoggingConfig schema
    :param pipeline_identifier: An optional identifier of the pipeline

    :return: A pipeline logger instance
    """
    if os.path.isfile(config):
        config = open(config)
    config = yaml.safe_load(config)
    config = PipelineLoggingConfig(**config)

    logger = build_logger(
        system_logging_config=config.system_logging,
        loggers_config=config.loggers,
        data_logging_config=possibly_modify_target_identifiers(
            config.data_logging, pipeline_identifier
        ),
    )
    return logger


def build_logger(
    system_logging_config: SystemLoggingConfig,
    data_logging_config: Optional[Dict[str, List[MetricFunctionConfig]]] = None,
    loggers_config: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> BaseLogger:
    """
    Builds a DeepSparse logger from the set of provided configs

    The process follows the following hierarchy:
        First: if global logger config is provided, the "leaf" loggers
            are built. Leaf loggers are the final loggers that log
            information to the final destination.

        Second: if data logging config is specified, a set of
            function loggers, responsible for data logging functionality
            wraps around the appropriate "leaf" loggers.

        Third: if system logging config is specified, a set of
            function loggers, responsible for system logging functionality
            wraps around the appropriate "leaf" loggers.

        Fourth: The resulting data and system loggers are wrapped inside a
            MultiLogger. Finally, the MultiLogger is wrapped inside
            an AsyncLogger to ensure that the logging process is asynchronous.

    For example:

    ```
    AsyncLogger:
      MultiLogger:
        FunctionLogger (system logging):
          target_logger:
            MultiLogger:
              LeafLogger1
              LeafLogger2
        FunctionLogger (system logging):
          ...
        FunctionLogger (data logging):
          target_logger:
            LeafLogger1
    ```

    :param system_logging_config: A SystemLoggingConfig instance that describes
        the system logging configuration
    :param data_logging_config: An optional dictionary that maps target names to
        lists of MetricFunctionConfigs.
    :param loggers_config: An optional dictionary that maps logger names to
        a dictionary of logger arguments.
    :return: a DeepSparseLogger instance
    """

    leaf_loggers = (
        build_leaf_loggers(loggers_config) if loggers_config else default_logger()
    )

    function_loggers_data = build_data_loggers(leaf_loggers, data_logging_config)
    function_loggers_system = build_system_loggers(leaf_loggers, system_logging_config)
    function_loggers = function_loggers_data + function_loggers_system

    return AsyncLogger(
        logger=MultiLogger(function_loggers),  # wrap all loggers to async log call
        max_workers=1,
    )


def get_target_identifier(
    target_name: str, pipeline_identifier: Optional[str] = None
) -> str:
    """
    Get the target identifier given the target name and a pipeline identifier

    :param target_name: The target name, can be a string or a regex pattern
    :param pipeline_identifier: Optional pipeline identifier. By default, is None
    :return: Final target identifier
    """
    if target_name.startswith("re:"):
        # if target name starts with "re:", it is a regex,
        # and we don't need to add the endpoint name to it
        return target_name
    if pipeline_identifier:
        # if pipeline_identifier specified,
        # prepend it to the target name
        return f"{pipeline_identifier}/{target_name}"
    return target_name


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
    loggers: Dict[str, BaseLogger],
    data_logging_config: Optional[
        Dict[str, List[MetricFunctionConfig]]
    ] = None,  # noqa F821
) -> List[FunctionLogger]:
    """
    Build a set of data loggers (FunctionLogger instances)
    according to the specified configuration.

    :param loggers: The created "leaf" loggers
    :param data_logging_config: The configuration of the data loggers.
        Specified as a dictionary that maps a target name to a list of metric functions.
    :return: A list of FunctionLogger instances responsible
        for logging data information
    """
    data_loggers = []
    if not data_logging_config:
        return data_loggers
    for target_identifier, metric_functions in data_logging_config.items():
        for metric_function in metric_functions:
            data_loggers.append(
                _build_function_logger(metric_function, target_identifier, loggers)
            )
    return data_loggers


def build_system_loggers(
    loggers: Dict[str, BaseLogger], system_logging_config: SystemLoggingConfig
) -> List[FunctionLogger]:
    """
    Build a set of  system loggers (FunctionLogger instances)
    according to the specified configuration.

    :param loggers: The created "leaf" loggers
    :param system_logging_config: The configuration of the system loggers.
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


def possibly_modify_target_identifiers(
    data_logging_config: Optional[Dict[str, List[MetricFunctionConfig]]] = None,
    pipeline_identifier: str = None,
) -> Optional[Dict[str, List[MetricFunctionConfig]]]:
    """
    Modify the target identifiers in the data logging config, given
    the specified pipeline identifier.

    :param data_logging_config: The configuration of the data loggers.
        Specified as a dictionary that maps a target name to a list of metric functions.
    :param pipeline_identifier: An optional string, that specifies
        the name of the pipeline the logging is being performed for.
    :return: the modified data_logging_config
    """
    if not data_logging_config or not pipeline_identifier:
        # if either of the arguments is None, return the original config
        return data_logging_config

    for target_identifier, metric_functions in data_logging_config.copy().items():
        if not target_identifier.startswith(pipeline_identifier):
            # if the target identifier does not already start
            # with the pipeline identifier, call get_target_identifier
            # to prepend it
            new_target_identifier = get_target_identifier(
                target_identifier, pipeline_identifier
            )
            data_logging_config[new_target_identifier] = data_logging_config.pop(
                target_identifier
            )
    return data_logging_config


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
