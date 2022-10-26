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

from collections import defaultdict

import numpy

import torch


class PythonLogger:
    pass


class PrometheusLogger:
    pass


class FunctionLogger:
    pass


class MultiLogger:
    pass


class SubprocessLogger:
    pass


def build_logger(server_config: "ServerConfig") -> "BaseLogger":
    """
    Hierarchy
    Leaf Loggers --> Multi Logger --> FunctionLogger --> SubprocessLogger
    """
    loggers_config = server_config.loggers
    functions_config = _extract_functions_config(server_config)
    functions_config = _substitute_func_str_for_callables(functions_config)
    if loggers_config is None:
        # will figure out whether not having loggers in config
        # results in some default logger (like PythonLogger)
        # or just None
        raise NotImplementedError()
    else:
        loggers = []
        for logger_name, logger_arguments in loggers_config.items():
            loggers.append(
                _build_single_logger(
                    logger_name=logger_name, logger_arguments=logger_arguments
                )
            )

        multi_logger = MultiLogger(loggers) if len(loggers) > 1 else loggers[0]
        function_logger = FunctionLogger(logger=multi_logger, config=functions_config)
        subprocess_logger = SubprocessLogger(logger=function_logger)

        return subprocess_logger


def _build_single_logger(logger_name, logger_arguments):
    """
    in future if logger_name == "prometheus":
        return PrometheusLogger(**logger_arguments)
    etc
    """
    return PythonLogger()


def _potentially_wrap_inside_function_logger(logger, functions_config):
    if isinstance(logger, PrometheusLogger):
        return FunctionLogger(logger=logger, config=functions_config)
    elif isinstance(logger, PythonLogger):
        return FunctionLogger(logger=logger, config=functions_config)
    else:
        return logger


def _extract_functions_config(server_config):
    functions_config = defaultdict(lambda: defaultdict(str))
    endpoints = server_config.endpoints
    for endpoint in endpoints:
        name = endpoint.name or endpoint.task
        if functions_config.get(name) is not None:
            raise ValueError()
        if endpoint.data_logging is not None:
            functions_config[name] = endpoint.data_logging

    return functions_config


def _substitute_func_str_for_callables(functions_config):
    for endpoint, data_logging_config in functions_config.items():
        for target, target_logging_config in data_logging_config.items():
            for function_config in target_logging_config:
                function_config["func"] = _parse_func_name(name=function_config["func"])

    return functions_config


def _parse_func_name(name: str):  # -> Callable[[np.array], np.array]:
    if name.startswith("torch."):
        return getattr(torch, name.split(".")[1])
    if name.startswith("numpy."):
        return getattr(numpy, name.split(".")[1])
    if name.startswith("builtins:"):
        return name
        # return getattr(<our_builtins module>, name.split(".")[1])
    # assume its a dynamic import function of the form '<path>:<name>'
    # path, func_name = name.split(":")
    # module = dynamic_import(path)
    # return getattr(module, func_name)
    return name


# def _build_logger_from_parsed_config(logger_param_config):
#    pass
