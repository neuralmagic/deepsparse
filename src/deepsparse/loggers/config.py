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

from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field, validator

from deepsparse.loggers.helpers import get_function_and_function_name


__all__ = [
    "MetricFunctionConfig",
]


class MetricFunctionConfig(BaseModel):
    """
    Holds logging configuration for a metric function
    """

    function: Callable[[Any], Any] = Field(
        description="The metric function callable. "
        "Used for mapping from the raw value from the pipeline to a value being logged."
    )
    function_name: str = Field(
        description="Name of the metric function. "
        "Is used for consistent naming of the metric and"
        "general book-keeping."
    )
    frequency: int = Field(
        description="Specifies how often the function should be applied"
        "(measured in numbers of inference calls).",
        default=1,
    )

    loggers: Optional[List[str]] = Field(
        default=None,
        description="Overrides the global logger configuration in "
        "the context of the DeepSparse server. "
        "If not None, this configuration stops logging data "
        "to globally specified loggers, and will only use "
        "the subset of loggers (specified here by a list of their names).",
    )

    @classmethod
    def from_server(
        cls, func: str, frequency: int = 1, loggers: Optional[List[str]] = None
    ) -> "MetricFunctionConfig":
        # automatically extract function and function name
        # from the function_identifier
        function, function_name = get_function_and_function_name(func)
        return cls(
            function=function,
            function_name=function_name,
            frequency=frequency,
            loggers=loggers,
        )

    @validator("frequency")
    def non_zero_frequency(cls, frequency: int) -> int:
        if frequency <= 0:
            raise ValueError(
                f"Passed frequency: {frequency}, but "
                "frequency must be a positive integer greater equal 1"
            )
        return frequency
