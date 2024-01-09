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
from typing import Any, Dict, Optional, Union

from .async_logger import AsyncLogger
from .config import LoggingConfig
from .logging_factory import LoggingConfigFactory


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogType(Enum):
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    METRICS = "METRICS"
    PROMETHEUS = "PROMETHEUS"


class SystemLogger:
    # Python logger
    def __init__(self, config: Dict[str, Any]):

        self.config = config
        factory = LoggingConfigFactory(config)
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


class PerformanceLogger(AsyncLogger):
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.Logger(__name__)  # temp

    def log(
        self,
        message: str,
    ):
        self.log(message)


class MetricsLogger(AsyncLogger):
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.Logger(__name__)  # temp

    def log(self, message: str):
        self.log(message)


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
        self.loggers = {
            LogType.SYSTEM: SystemLogger(self.config.get("system")),
            LogType.PERFORMANCE: PerformanceLogger(self.config.get("performance")),
            LogType.METRICS: MetricsLogger(self.config.get("metrics")),
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
