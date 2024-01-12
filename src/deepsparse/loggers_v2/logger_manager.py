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

from concurrent.futures import Future
from enum import Enum
from typing import Any, Dict, Optional

from .async_submitter import AsyncSubmitter
from .config import LoggingConfig
from .logger_factory import (
    MetricLogger,
    PerformanceLogger,
    SystemLogger,
    instantiate_logger,
    logger_factory,
)


class LogType(Enum):
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    METRIC = "METRIC"


class LoggerManager(AsyncSubmitter):
    # class LoggerManager(AsyncSubmitter, FrequencyManager, LoggerFactory):

    """
    Initialize loggers for Pipeline

    :param config: Path to yaml or stringified yaml
    """

    def __init__(self, config: Optional[str] = None):
        super().__init__()
        self.config = LoggingConfig.from_config(config).dict()
        self.leaf_logger: Dict = {}
        self.instantiate_leaf_logger(self.config.get("logger"))

        # send logggers the factory with the config
        factory = logger_factory(
            self.config,
            self.leaf_logger,
        )

        self.loggers = {
            LogType.SYSTEM: factory.get("system"),
            LogType.PERFORMANCE: factory.get("performance"),
            LogType.METRIC: factory.get("metric"),
        }

    def log(
        self,
        *args,
        **kwargs,
    ):
        self.submit(
            self.run,
            self.callback,
            *args,
            **kwargs,
        )

    def run(
        self, log_type: str, value: Any, tag: Optional[str] = None, *args, **kwargs
    ):
        log_type = log_type.upper()
        if log_type in LogType.__members__:
            logger = self.loggers.get(LogType[log_type])
            if logger:
                logger.log(value=value, tag=tag, *args, **kwargs)

    def callback(self, future: Future):
        exception = future.exception()
        if exception is not None:
            self.system.log(
                f"Exception occurred during async logging job: {repr(exception)}",
                level="error",
            )

    def instantiate_leaf_logger(self, logger_config: Dict[str, Dict]):
        for name, init_args in logger_config.items():
            self.leaf_logger[name] = instantiate_logger(
                name=init_args.pop("name"),
                init_args=init_args,
            )

    @property
    def system(self) -> Optional[SystemLogger]:
        return self.loggers[LogType.SYSTEM]

    @property
    def performance(self) -> Optional[PerformanceLogger]:
        return self.loggers[LogType.PERFORMANCE]

    @property
    def metric(self) -> Optional[MetricLogger]:
        return self.loggers[LogType.METRIC]
