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
from pydantic import BaseModel, Extra, Field, validator


class LogLevelEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERRPR"
    CRITICAL = "CRITICAL"


class StreamLoggingConfig(BaseModel):
    level: str = Field(default="INFO", description="Logger level")
    formatter: str = Field(
        default="%(asctime)s - %(levelname)s - %(message)s",
        description="Log display format",
    )


class FileLoggingConfig(StreamLoggingConfig):
    filename: str = Field(
        default="/tmp/pipeline.log", description="Path to save the logs"
    )


class RotatingLoggingConfig(StreamLoggingConfig):
    filename: str = Field(
        default="/tmp/pipeline_rotate.log", description="Path to save the logs"
    )
    max_bytes: int = Field(default=2048, description="Max size till rotation")
    backup_count: int = Field(default=3, description="Number of backups")


class PerformanceConfig(BaseModel):
    enabled: bool = Field(default=True, description="True to log, False to ignore")
    frequency: int = Field(
        default=1, description="The rate to log. Log every N occurances"
    )
    loggers: List[str] = Field(
        default=["python"],
        description=(
            "List of loggers to use. Should be in the format",
            "path/to/file.py:ClassName",
        ),
    )


class PerformanceLoggingConfig(BaseModel):
    __root__: Optional[Dict[str, PerformanceConfig]]


# class MetricConfig(PerformanceConfig):
#     func: str = Field(
#         default=None,
#         description="Callable to use for logging value",
#     )


class MetricLoggingConfig(BaseModel):
    # __root__: Optional[Dict[str, PerformanceConfig]]
    __root__: Optional[Dict]


class PythonLoggerConfig(BaseModel):
    # level: str = Field(default="INFO", description="Root logger level")
    stream: StreamLoggingConfig = Field(
        default=StreamLoggingConfig(), description="Stream logging config"
    )
    file: FileLoggingConfig = Field(
        default=FileLoggingConfig(), description="File logging config"
    )
    rotating: RotatingLoggingConfig = Field(
        default=RotatingLoggingConfig(), description="Rotating logging config"
    )
    
class LoggingHandlerConfig(BaseModel):
    level: str = Field(default="INFO", description="Root logger level")
    stream: StreamLoggingConfig = Field(
        default=StreamLoggingConfig(), description="Stream logging config"
    )
    file: FileLoggingConfig = Field(
        default=FileLoggingConfig(), description="File logging config"
    )
    rotating: RotatingLoggingConfig = Field(
        default=RotatingLoggingConfig(), description="Rotating logging config"
    )
    
    


class CustomLoggingConfig(BaseModel):
    frequency: int = Field(
        default=1, description="The rate to log. Log every N occurances"
    )
    use: str = Field(
        description=(
            "List of loggers to use. Should be in the format",
            "path/to/file.py:ClassName",
        ),
    )
    handler: Optional[LoggingHandlerConfig] = Field(
        default=None,
        description="Python logging handler config",
    )
    class Config:
        extra = Extra.allow  # Allow extra kwargs



class PrometheusLoggerConfig(BaseModel):
    alias: Optional[str]
    port: int
    filename: str


class LoggerConfig(BaseModel):
    __root__: Optional[Dict]


class SystemTargetConfig(BaseModel):
    tag: Optional[List[str]] = Field(None, description="Tag id to register logging")
    func: List[str] = Field(
        "identity",
        description="Callable to apply to 'value' for logging. Defaults to ",
    )


class MetricTargetConfig(SystemTargetConfig):
    name: List[str] = Field(
        None, description="Name of a desired ClassName.__class__.__name__ to log"
    )
    output_key: List[str] = Field(
        None,
        description="If the callable output of ClassName is a dict, then log the value from the key output_key.",
    )



class LoggingConfig(BaseModel):

    version: int = Field(
        deafult=2,
        description="Pipeline logger version",
    )

    target: Dict[str, Union[SystemTargetConfig, MetricTargetConfig]]
    
    logger: Dict[str, Union{CustomLoggingConfig, PrometheusLoggingConfig, PythonLoggingConfig}]

    @validator("target", pre=True, always=True)
    def validate_target(cls, value):
        validated_target = {}
        for key, config in value.items():
            if "name" in config and "output_key" in config:
                validated_target[key] = MetricTargetConfig(**config)
            else:
                validated_target[key] = SystemTargetConfig(**config)
        return validated_target


    # system: str = Field(
    #     default="python",
    #     description="Default python logging module logger",
    # )

    # performance: PerformanceLoggingConfig = Field(
    #     default=PerformanceLoggingConfig(),
    #     description="System level config",
    # )

    # metric: MetricLoggingConfig = Field(
    #     default=MetricLoggingConfig(),
    #     description="Metrics configuration",
    # )

    # @validator("logger", pre=True, always=True)
    # def validate_logger(cls, value):
    #     validated_logger = {}
    #     if "python" not in value:
    #         value["python"] = {}

    #     for key, config in value.items():
    #         if key == "prometheus":
    #             validated_logger[key] = PrometheusLoggerConfig(
    #                 **cls.set_alias(config, key)
    #             )
    #         elif key == "python":
    #             validated_logger[key] = PythonLoggerConfig(**cls.set_alias(config, key))
    #         else:
    #             validated_logger[key] = CustomLoggerConfig(**cls.set_alias(config, key))
    #     return validated_logger

    # @staticmethod
    # def set_alias(config, key):
    #     if "alias" not in config:
    #         if key == "python":
    #             config["config"] = "python"
    #             return config
    #         if key == "prometheus":
    #             config["alias"] = "prometheus"
    #             return config
    #         if ":" in key:
    #             alias = key.split(":")[-1]
    #             config["alias"] = alias
    #             return config

    #         config["alias"] = key
    #     return config

    # @validator("performance", pre=True, always=True)
    # def validate_performance(cls, value: Dict, values: Dict):
    #     """
    #     Validate performance loggers are a subset of
    #         loggers defined in root level loggers in config
    #         and validate perfoamnce fields

    #     :param value: Performance config
    #     :param values: All values in LoggingConfig

    #     """

    #     cls.validate_loggers_subset(value, values)

    #     validated_performance = {}
    #     for key, config in value.items():
    #         validated_performance[key] = PerformanceConfig(**config)

    #     return validated_performance

    # @classmethod
    # def validate_loggers_subset(cls, value: Dict, values: Dict):
    #     """
    #     Check that performance loggers are set in root level
    #     logger

    #     :param value: Performance config
    #     :param values: All values in LoggingConfig

    #     """
    #     performance_loggers = {
    #         logger for config in value.values() for logger in config.get("loggers", [])
    #     }

    #     aliases = {
    #         value["alias"]
    #         for value in values["logger"].dict().get("__root__", {}).values()
    #         if "alias" in value
    #     }

    #     for performance_logger in performance_loggers:
    #         if performance_logger not in aliases:
    #             raise ValueError(
    #                 f"Performance loggers must be a subset of {aliases}. Got {performance_logger}"
    #             )

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
    def from_config(cls, config: str):
        """Helper to load from file or string"""
        if config.endswith(".yaml"):
            return cls.from_yaml(config)
        return cls.from_str(config)
