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


class PythonLoggingConfig(BaseModel):
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

    class Config:
        extra = Extra.allow  # Allow extra kwargs


class PrometheusLoggingConfig(BaseModel):
    use: str = Field(default="path", description="Prometheus Logging path")
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

    logger: Dict[
        str, Union[CustomLoggingConfig, PrometheusLoggingConfig, PythonLoggingConfig]
    ]

    system: str = Field(
        default="python",
        description="Default python logging module logger",
    )

    performance: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Performance level config",
    )

    metric: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Metric level config",
    )

    @validator("target", pre=True, always=True)
    def validate_target(cls, value):
        validated_target = {}
        for key, config in value.items():
            if "name" in config and "output_key" in config:
                validated_target[key] = MetricTargetConfig(**config)
            else:
                validated_target[key] = SystemTargetConfig(**config)
        return validated_target

    @validator("system", pre=True, always=True)
    def validate_system(cls, value: Dict, values: Dict):
        """
        Validate system loggers are a subset of
            loggers defined in root level loggers in config

        :param value: system config
        :param values: All values in LoggingConfig
        """
        cls.validate_loggers_subset([value], values)
        return value

    @validator("performance", pre=True, always=True)
    def validate_performance(cls, value: Dict, values: Dict):
        """
        Validate performance loggers are a subset of
            loggers defined in root level loggers in config
        Validate that targets are a subset of
            targets defined in the root level
            and

        :param value: performance config
        :param values: All values in LoggingConfig
        """
        loggers = list(value.keys())
        targets = list(set(target for sublist in value.values() for target in sublist))

        cls.validate_loggers_subset(loggers, values)
        cls.validate_targets_subset(targets, values)
        return value

    @validator("metric", pre=True, always=True)
    def validate_metric(cls, value: Dict, values: Dict):
        """
        Validate metric loggers are a subset of
            loggers defined in root level loggers in config
        Validate that targets are a subset of
            targets defined in the root level
            and

        :param value: metric config
        :param values: All values in LoggingConfig
        """
        loggers = list(value.keys())
        targets = list(set(target for sublist in value.values() for target in sublist))

        cls.validate_loggers_subset(loggers, values)
        cls.validate_targets_subset(targets, values)
        cls.validate_metric_target_fields(targets, values)

        return value

    @classmethod
    def validate_metric_target_fields(cls, targets: List[str], values: Dict):
        root_target = values["target"]
        for target in targets:
            if not isinstance(root_target.get(target), MetricTargetConfig):
                raise ValueError(
                    f"Defined target {target} must have name and output_key fields."
                )

    @classmethod
    def validate_targets_subset(cls, targets: List[str], values: Dict):
        """
        Check that targets are a subset of the root level
            target

        :param value: Performance config
        :param values: All values in LoggingConfig

        """
        targets = {target_id for target_id in values["target"].keys()}
        for target in targets:
            if target not in targets:
                raise ValueError(
                    f"Defined target {target} must be a subset of {targets}."
                )

    @classmethod
    def validate_loggers_subset(cls, loggers: List[str], values: Dict):
        """
        Check that loggers are a subset of the root level
            logger

        :param value: Performance config
        :param values: All values in LoggingConfig

        """
        loggers = {logger_id for logger_id in values["logger"].keys()}
        for logger in loggers:
            if logger not in loggers:
                raise ValueError(
                    f"Defined logger {logger} must be a subset of {loggers}."
                )

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
        # """Helper to load from file or string"""
        # if config.endswith(".yaml"):
        #     return cls.from_yaml(config)
        # return cls.from_str(config)
        return yaml.safe_load(config)
