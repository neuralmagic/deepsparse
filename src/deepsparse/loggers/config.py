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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy
from pydantic import BaseModel, Field, root_validator, validator

import deepsparse.loggers.metric_functions.built_ins as built_ins
import torch


__all__ = [
    "MetricFunctionConfig",
    "TargetLoggingConfig",
]


def get_function_and_function_name(
    function_identifier: str,
) -> Tuple[Callable[[Any], Any], str]:
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

    func: str = Field(description="Identifier of the metric function")
    function: Callable[[Any], Any] = Field(description="The metric function callable")
    function_name: str = Field(description="Name of the metric function")
    frequency: int = Field(
        default=1,
        description="Specifies how often the function should be applied"
        "(measured in numbers of inference calls",
    )
    logger: Optional[List[str]] = Field(
        default=None,
        description="Overrides the global logger configuration in "
        "the context of the DeepSparse server. "
        "If not None, this configuration stops logging data "
        "to globally specified loggers, and will only use "
        "the subset of loggers "
        "(specified by a list of their names)",
    )

    def __init__(self, **data):
        if data.get("function"):
            raise ValueError()
        if data.get("function_name"):
            raise ValueError()
        function_identifier = data.get("func")
        # automatically extract function and function name
        # from the function_identifier
        function, function_name = get_function_and_function_name(function_identifier)
        data["function"], data["function_name"] = function, function_name
        super().__init__(**data)

    @validator("frequency")
    def non_zero_frequency(cls, frequency):
        if frequency <= 0:
            raise ValueError(
                f"Passed frequency: {frequency}, but "
                "frequency must be a positive integer greater equal 1"
            )
        return frequency


class TargetLoggingConfig(BaseModel):
    """
    Holds configuration for a single data logging target
    """

    target: str = Field(description="Name of the target.")
    functions: List[MetricFunctionConfig] = Field(
        description="List of MetricFunctionConfigs pertaining to the target"
    )

    @root_validator
    def unique_mapping_names_per_endpoint(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make sure that for each target, two mappings with the same name
        are not applied.
        """
        target, functions = data["target"], data["functions"]
        potential_duplicates = set()
        for functions_count, cfg in enumerate(functions):
            potential_duplicates.add(cfg.function_name)
            if len(potential_duplicates) != functions_count + 1:
                raise ValueError(
                    f"For target - {target} - found multiple "
                    "metric functions with the same "
                    f"name: {cfg.function_name}. Make sure "
                    f"that that there are no duplicated metric "
                    f"functions being applied to the same target."
                )
        return data
