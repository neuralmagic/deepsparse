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
from typing import Any, Dict, List, Optional, Type

from deepsparse import Pipeline
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
    "add_logger_to_pipeline",
    "build_logger",
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


def add_logger_to_pipeline(
    config: PipelineLoggingConfig, pipeline: Pipeline
) -> Pipeline:
    """
    Add a logger to the pipeline according to the configuration

    :param config: The configuration of the pipeline logger
    :param pipeline: The pipeline to add the logger to
    :return: The pipeline with the logger added
    """
    if config.data_logging:
        for target, metric_functions in config.data_logging.copy().items():
            # modify the base target name if required
            new_target = get_target_identifier(
                target_name=target, pipeline_identifier=pipeline._identifier()
            )
            if not new_target == target:
                # if the target name is altered, we need to update the config
                config.data_logging[new_target] = metric_functions
                del config.data_logging[target]

    logger = build_logger(
        system_logging_config=config.system_logging,
        loggers_config=config.loggers,
        data_logging_config=config.data_logging,
    )
    pipeline.logger = logger
    return pipeline


def build_logger(
    system_logging_config: SystemLoggingConfig,
    data_logging_config: Optional[Dict[str, List[MetricFunctionConfig]]] = None,
    loggers_config: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> BaseLogger:
    """
    A general function for building a logger instance according to the specified
    configuration

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
