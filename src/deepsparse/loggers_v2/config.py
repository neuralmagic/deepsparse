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
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Extra, Field, root_validator, validator


# class LogLevelEnum(str, Enum):
#     DEBUG = "DEBUG"
#     INFO = "INFO"
#     WARNING = "WARNING"
#     ERROR = "ERRPR"
#     CRITICAL = "CRITICAL"


# class StreamLoggingConfig(BaseModel):
#     level: str = Field(default="INFO", description="Logger level")
#     formatter: str = Field(
#         default="%(asctime)s - %(levelname)s - %(message)s",
#         description="Log display format",
#     )


# class FileLoggingConfig(StreamLoggingConfig):
#     filename: str = Field(
#         default="/tmp/pipeline.log", description="Path to save the logs"
#     )


# class RotatingLoggingConfig(StreamLoggingConfig):
#     filename: str = Field(
#         default="/tmp/pipeline_rotate.log", description="Path to save the logs"
#     )
#     max_bytes: int = Field(default=2048, description="Max size till rotation")
#     backup_count: int = Field(default=3, description="Number of backups")


# class PythonLoggingConfig(BaseModel):
#     level: str = Field(default="INFO", description="Root logger level")
#     stream: StreamLoggingConfig = Field(
#         default=StreamLoggingConfig(), description="Stream logging config"
#     )
#     file: FileLoggingConfig = Field(
#         default=FileLoggingConfig(), description="File logging config"
#     )
#     rotating: RotatingLoggingConfig = Field(
#         default=RotatingLoggingConfig(), description="Rotating logging config"
#     )


# class CustomLoggingConfig(BaseModel):
#     frequency: int = Field(
#         default=1, description="The rate to log. Log every N occurances"
#     )
#     use: str = Field(
#         description=(
#             "List of loggers to use. Should be in the format",
#             "path/to/file.py:ClassName",
#         ),
#     )

#     class Config:
#         extra = Extra.allow  # Allow extra kwargs


# class PrometheusLoggingConfig(BaseModel):
#     use: str = Field(default="path", description="Prometheus Logging path")
#     port: int
#     filename: str


class LoggerConfig(BaseModel):
    name: str = Field(
        default="PythonLogger",
        description="Path (/path/to/file:FooLogger) or name of loggers in deepsparse/loggers/registry/__init__ path",
    )
    handler: Optional[Dict] = None


class TargetConfig(BaseModel):
    func: str = Field(
        default="identity",
        description=(
            "Callable to apply to 'value' for dimensionality reduction, if necessary"
            "func can be a path /path/to/file:func) or name of func in deepsparse/loggers/registry/__init__ path"
        ),
    )

    freq: int = Field(
        default=1,
        description="The rate to log. Log every N occurances",
    )
    uses: List[str] = Field(default=["default"], description="")


class MetricTargetConfig(TargetConfig):
    capture: List[str] = Field(
        ["*"],
        description="Key of the output dict. Corresponding value will be logged. The value can be a regex pattern",
    )


class LoggingConfig(BaseModel):

    version: int = Field(
        default=2,
        description="Pipeline logger version",
    )

    logger: Dict[str, LoggerConfig] = Field(
        default=dict(default=LoggerConfig()),
        description="Loggers to be Used",
    )

    system: Dict[str, List[TargetConfig]] = Field(
        default={"*": [TargetConfig()]},
        description="Default python logging module logger",
    )

    performance: Dict[str, List[TargetConfig]] = Field(
        default={"cpu": [TargetConfig()]},
        description="Performance level config",
    )

    metric: Dict[str, List[MetricTargetConfig]] = Field(
        default={r"(?i)operator": [MetricTargetConfig()]},
        description="Metric level config",
    )
    
    @validator("logger", always=True)
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
        # """Helper to load from file or string"""
        if config:
            if config.endswith(".yaml"):
                return cls.from_yaml(config)
            return cls.from_str(config)
        return LoggingConfig()
