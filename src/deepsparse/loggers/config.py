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

from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, validator


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(
        default="PythonLogger",
        description=(
            "Path (/path/to/file:FooLogger) or name of loggers in "
            "deepsparse/loggers/registry/__init__ path"
        ),
    )
    handler: Optional[Dict] = None


class TargetConfig(BaseModel):
    func: str = Field(
        default="identity",
        description=(
            (
                "Callable to apply to 'value' for dimensionality reduction. "
                "func can be a path /path/to/file:func) or name of func in "
                "deepsparse/loggers/registry/__init__ path"
            )
        ),
    )

    freq: int = Field(
        default=1,
        description="The rate to log. Log every N occurances",
    )
    uses: List[str] = Field(default=["default"], description="")


class MetricTargetConfig(TargetConfig):
    capture: Optional[List[str]] = Field(
        None,
        description=(
            "Key of the output dict. Corresponding value will be logged. "
            "The value can be a regex pattern"
        ),
    )


class LoggingConfig(BaseModel):

    loggers: Dict[str, LoggerConfig] = Field(
        default=dict(default=LoggerConfig()),
        description="Loggers to be Used",
    )

    system: Dict[str, List[TargetConfig]] = Field(
        default={"re:.*": [TargetConfig()]},
        description="Default python logging module logger",
    )

    performance: Dict[str, List[TargetConfig]] = Field(
        default={"cpu": [TargetConfig()]},
        description="Performance level config",
    )

    metric: Dict[str, List[MetricTargetConfig]] = Field(
        default={"re:(?i)operator": [MetricTargetConfig()]},
        description="Metric level config",
    )

    @validator("loggers", always=True)
    def always_include_python_logger(cls, value):
        if "default" not in value:
            value["default"] = LoggerConfig()
        return value

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load from yaml file"""
        with open(yaml_path, "r") as file:
            yaml_content = yaml.safe_load(file)
        return cls(**yaml_content)

    @classmethod
    def from_str(cls, stringified_yaml: str):
        """Load from stringified yaml"""
        yaml_content = yaml.safe_load(stringified_yaml)

        return cls(**yaml_content)

    @classmethod
    def from_config(cls, config: Optional[str] = None):
        """Helper to load from file or string"""
        if config:
            if config.endswith(".yaml"):
                return cls.from_yaml(config)
            return cls.from_str(config)
        return LoggingConfig()
