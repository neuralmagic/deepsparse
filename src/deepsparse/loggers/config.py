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
Pydantic Models for Logging Configs
"""
import importlib
from typing import Any, List, Optional

import numpy
from pydantic import BaseModel, Field

import deepsparse.loggers.metric_functions.built_ins as built_ins
import torch


__all__ = ["MetricFunctionConfig", "TargetLoggingConfig", "PipelineLoggingConfig"]


def get_function_and_function_name(
    function_identifier: str,
):  # -> Callable[[np.array], np.array]:
    """
    Parse function identifier and return the function as well as its name

    :param function_identifier: Can be one of the following:

        1. framework function, e.g.
            "torch.mean" or "numpy.max"

        2. built-in function, e.g.
            "builtins:function_name"

        3. user-defined function in the form of
           '<path_to_the_python_script>:<function_name>', e.g.
           "{...}/script_name.py:function_name"

    :return: A tuple (function, function name)
    """
    # TODO: Add return type
    if function_identifier.startswith("torch."):
        func_name = function_identifier.split(".")[1]
        return getattr(torch, func_name), func_name
    if function_identifier.startswith("numpy.") or function_identifier.startswith(
        "np."
    ):
        func_name = function_identifier.split(".")[1]
        return getattr(numpy, func_name), func_name
    if function_identifier.startswith("builtins:"):
        func_name = function_identifier.split(":")[1]
        return getattr(built_ins, func_name), func_name
    # assume a dynamic import function of the form
    # '<path_to_the_python_script>:<function_name>'
    path, func_name = function_identifier.split(":")
    spec = importlib.util.spec_from_file_location("user_defined_metric_functions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name), func_name


class MetricFunctionConfig(BaseModel):
    """
    Holds logging configuration for a metric function
    """

    func: Any = Field(description="Metric function object")  # TODO: Specify it
    function_name: str = Field(description="Name of the metric function")
    frequency: int = Field(
        description="Specifies how often the function should be applied"
    )

    def __init__(self, **data):
        function_identifier = data.get("func")
        # automatically extract function and function name
        # from the function_identifier
        func, function_name = get_function_and_function_name(function_identifier)
        data["func"] = func
        data["function_name"] = function_name
        super().__init__(**data)


class TargetLoggingConfig(BaseModel):
    """
    Holds logging configuration for a target
    """

    target: str = Field(description="Name of the target.")
    mappings: List[MetricFunctionConfig] = Field(
        description="List of MetricFunctionConfigs pertaining to the target"
    )


class PipelineLoggingConfig(BaseModel):
    """
    Holds logging configuration for a single pipeline/endpoint
    """

    name: Optional[str] = Field(
        default=None, description="Name of the pipeline/endpoint."
    )
    targets: List[TargetLoggingConfig] = Field(
        description="List of TargetLoggingConfigs pertaining to the pipeline/endpoint"
    )


class MultiplePipelinesLoggingConfig(BaseModel):
    """
    Holds multiple PipelineLoggingConfigs
    """

    pipelines: List[PipelineLoggingConfig]
