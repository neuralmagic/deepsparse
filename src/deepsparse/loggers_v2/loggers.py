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

import logging
from enum import Enum
from typing import Dict, Optional, Union

# from deepsparse.loggers_v2.config import LoggingConfig


class LogType(Enum):
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    METRICS = "METRICS"


class Logger:
    def __init__(self):
        # self.logger = None
        ...

    def log(self, *arg, **kwarg):
        ...


class SystemLogger(Logger):
    class LogLevelEnum(str, Enum):
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    def __init__(self):
        self.loggers = {log_type: self.create_logger(log_type) for log_type in LogType}

    def create_logger(self, log_type=LogLevelEnum):
        logger = logging.getLogger(f"SystemLogger.{log_type.name}")
        return logger

    def log(self, message: str, level: Optional[str] = "INFO"):
        logger = self.loggers.get(level)
        if logger:
            logger.info(message)


class PerformanceLogger(Logger):
    ...


class MetricsLogger(Logger):
    ...


# def parse(data: Union[str, Dict]):
#     if isin


class PipelineLogger:
    """
    Given logging config, sets up the logging object

    :param config: path to to the config or raw yaml
    """

    def __init__(self, config: str):
        # self.config = config
        if config.endswith(".yaml") or config.endswith(".yml"):
            import yaml

            with open(config, "r") as yaml_file:
                configs = yaml.safe_load(yaml_file)

        self._loggers = {
            LogType.SYSTEM: SystemLogger(),
            LogType.PERFORMANCE: PerformanceLogger(),
            LogType.METRICS: MetricsLogger(),
        }

    def log(self, message: str, type: str, tag: Optional[str] = None, **kwargs):
        logger = self._loggers.get(type)
        if logger is not None:
            logger.log(message, tag=tag, **kwargs)

    def __call__(self):
        ...

    def __repr__(self):
        ...
