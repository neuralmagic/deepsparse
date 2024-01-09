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
from typing import Any, Dict, List, Optional, Union

from .async_logger import AsyncLogger
from .config import LoggingConfig
from .logging_factory import PerformanceLoggerFactory, SystemLoggerFactory


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogType(Enum):
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    METRIC = "METRIC"
    PROMETHEUS = "PROMETHEUS"


class SystemLogger:
    # Python logger
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        factory = SystemLoggerFactory(config)
        self.logger = factory.create_logger()

    def log(
        self,
        value: Any,
        tag: Optional[str] = None,
        level: str = "info",
        *args,
        **kwargs,
    ):

        level = level.lower()
        attr = getattr(self.logger, level)
        msg = f"value={value}"
        if tag is not None:
            msg += f" tag={tag}"
        if args:
            msg += f" args={args}"
        if kwargs:
            msg += f" kwargs={kwargs}"

        attr(msg)


class PerformanceLogger:
    def __init__(self, config: Dict):
        self.config = config
        factory = PerformanceLoggerFactory(config)
        factory.create_logger()
        # self.logger: List = factory.logger
        self.logger = factory

    def log(
        self,
        *args,
        **kwargs,
    ):
        self.logger.log(*args, **kwargs)


class MetricsLogger(AsyncLogger):
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.Logger(__name__)  # temp

    def log(self, message: str):
        self.logger.log(message)


class PrometheusLogger:
    def __init__(self, config: Dict):
        ...


class PipelineLogger:
    """
    Initialize loggers for Pipeline

    :param config: Path to yaml or stringified yaml
    """

    def __init__(self, config: str):
        self.config = LoggingConfig.from_config(config).dict()
        breakpoint()
        self.loggers = {
            LogType.SYSTEM: SystemLogger(self.config.get("system")),
            LogType.PERFORMANCE: PerformanceLogger(self.config.get("performance")),
            LogType.METRIC: MetricsLogger(self.config.get("metrics")),
            LogType.PROMETHEUS: PrometheusLogger(self.config.get("prometheus")),
        }

    def log(
        self, log_type: str, value: Any, tag: Optional[str] = None, *args, **kwargs
    ):
        log_type = log_type.upper()
        if log_type in LogType.__members__:
            logger = self.loggers.get(LogType[log_type])
            if logger:
                logger.log(value=value, tag=tag, *args, **kwargs)
