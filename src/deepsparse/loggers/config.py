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

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


"""
Implements schemas for the configs pertaining to logging
"""

__all__ = [
    "MetricFunctionConfig",
    "SystemLoggingGroup",
    "SystemLoggingConfig",
    "PipelineLoggingConfig",
]


class MetricFunctionConfig(BaseModel):
    """
    Holds logging configuration for a metric function
    """

    func: str = Field(
        description="The name that specifies the metric function to be applied. "
        "It can be: "
        "1) a built-in function name "
        "2) a dynamic import function of the form "
        "'<path_to_the_python_script>:<function_name>' "
        "3) a framework function (e.g. np.mean or torch.mean)"
    )

    frequency: int = Field(
        description="Specifies how often the function should be applied"
        "(measured in numbers of inference calls).",
        default=1,
    )

    target_loggers: List[str] = Field(
        default=[],
        description="Overrides the global logger configuration."
        "If not an empty list, this configuration stops logging data "
        "to globally specified loggers, and will only use "
        "the subset of loggers (specified here by a list of their names).",
    )

    @validator("frequency")
    def non_zero_frequency(cls, frequency: int) -> int:
        if frequency <= 0:
            raise ValueError(
                f"Passed frequency: {frequency}, but "
                "frequency must be a positive integer greater equal 1"
            )
        return frequency


class SystemLoggingGroup(BaseModel):
    """
    Holds the configuration for a single system logging group.
    """

    enable: bool = Field(
        default=False,
        description="Whether to enable the system logging group. Defaults to False",
    )

    target_loggers: List[str] = Field(
        default=[],
        description="The list of target loggers to log to. "
        "If None, logs to all the available loggers",
    )


class SystemLoggingConfig(BaseModel):
    """
    Holds the configuration for the system logging
    in the context of a single pipeline
    """

    # Global Logging Config
    enable: bool = Field(
        default=True, description="Whether to enable system logging. Defaults to True"
    )

    # System Logging Groups
    resource_utilization: SystemLoggingGroup = Field(
        default=SystemLoggingGroup(enable=False),
        description="The configuration group for the resource "
        "utilization system logging group. By default this group is disabled.",
    )
    prediction_latency: SystemLoggingGroup = Field(
        default=SystemLoggingGroup(enable=True),
        description="The configuration group for the prediction latency "
        "system logging group. By default this group is enabled.",
    )


class PipelineLoggingConfig(BaseModel):
    """
    Holds the complete configuration for the logging
    in the context of a single pipeline
    """

    loggers: Dict[str, Optional[Dict[str, Any]]] = Field(
        default={},
        description=(
            "Optional dictionary of logger integration names to initialization kwargs."
            "Set to {} for no loggers. Default is {}."
        ),
    )

    system_logging: SystemLoggingConfig = Field(
        default=SystemLoggingConfig(),
        description="A model that holds the system logging configuration. "
        "If not specified explicitly in the yaml config, the "
        "default SystemLoggingConfig model is used.",
    )

    data_logging: Optional[Dict[str, List[MetricFunctionConfig]]] = Field(
        default=None,
        description="Specifies the rules for the data logging. "
        "It relates a key (name of the logging target) "
        "to a list of metric functions that are to be applied"
        "to this target prior to logging.",
    )
