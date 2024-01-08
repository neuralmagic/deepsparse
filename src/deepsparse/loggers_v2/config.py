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

from enum import Enum

import yaml
from pydantic import BaseModel, Extra, Field, validator


class LogLevelEnum(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemLoggingConfig(BaseModel):
    level: LogLevelEnum = Field(
        default=LogLevelEnum.INFO,
        description=(
            "Python logging level. Choose from"
            "'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'"
        ),
    )

    @validator("level", pre=True)
    def convert_to_lowercase(cls, value):
        return value.lower()


class PerformanceLoggingConfig(BaseModel):
    frequency: float = Field(
        default=1.0, description="The rate to save the logs when they are passed in"
    )
    timings: bool = True
    cpu: bool = True


class MetricConfig(BaseModel):
    function: str
    frequency: float


class MetricsConfig(BaseModel):
    __root__: dict[str, MetricConfig]

    @validator("__root__")
    def validate_metrics(cls, value):
        for op_name_and_key, metric_config in value.items():
            cls.validate_metric_name(op_name_and_key)  # Validate metric_name format
            MetricConfig.validate(
                metric_config
            )  # Validate each MetricConfig without creating an instance
        return value

    @classmethod
    def validate_metric_name(cls, op_name_and_key):
        parts = op_name_and_key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid metric name format: {op_name_and_key}."
                "Should be in the format 'op_name.op_key'."
            )


class LoggingConfig(BaseModel):
    system: SystemLoggingConfig = Field(
        default=SystemLoggingConfig(),
        description="System level config",
    )

    performance: PerformanceLoggingConfig = Field(
        default=PerformanceLoggingConfig(),
        description="Performance level config",
    )

    metrics: MetricsConfig = Field(
        default={},
        description="Metrics configuration",
        extra=Extra.allow,
    )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as file:
            yaml_content = yaml.safe_load(file)
        return cls(**yaml_content)

    @classmethod
    def from_str(cls, stringified_yaml: str):
        yaml_content = yaml.safe_load(stringified_yaml)
        return cls(**yaml_content)

        # if yaml_path.endswith(".yaml") or yaml_path.endswith(".yml"):
