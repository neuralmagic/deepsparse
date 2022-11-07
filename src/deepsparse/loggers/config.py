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
    "PipelineLoggingConfig",
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

    func: Callable[[Any], Any] = Field(description="Metric function object")
    function_name: str = Field(description="Name of the metric function")
    frequency: int = Field(
        default=1, description="Specifies how often the function should be applied"
    )
    logger: Optional[List[str]] = Field(default=None)

    def __init__(self, **data):
        function_identifier = data.get("func")
        # automatically extract function and function name
        # from the function_identifier
        func, function_name = get_function_and_function_name(function_identifier)
        data["func"], data["function_name"] = func, function_name
        super().__init__(**data)

    @validator("frequency")
    def name_must_contain_space(cls, frequency):
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

    mappings: List[MetricFunctionConfig] = Field(
        description="List of MetricFunctionConfigs pertaining to the target"
    )

    @root_validator
    def unique_mapping_names_per_endpoint(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make sure that for each target, two mappings with the same name
        are not applied.

        :param data:
        :return:
        """
        target, mappings = data["target"], data["mappings"]
        potential_duplicates = set()
        for mapping_count, mapping_cfg in enumerate(mappings):
            potential_duplicates.add(mapping_cfg.function_name)
            if len(potential_duplicates) != mapping_count + 1:
                raise ValueError(
                    f"For target - {target} - found multiple "
                    "metric functions with the same "
                    f"name: {mapping_cfg.function_name}. Make sure "
                    f"that that there are no duplicated metric "
                    f"functions being applied to the same target."
                )
        return data


class PipelineLoggingConfig(BaseModel):
    """
    Holds logging configuration for a single data logging pipeline/endpoint
    """

    name: Optional[str] = Field(
        default=None, description="Name of the pipeline/endpoint."
    )
    targets: List[TargetLoggingConfig] = Field(
        description="List of TargetLoggingConfigs pertaining to the pipeline/endpoint"
    )
