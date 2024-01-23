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
from typing import Any, Optional

from .async_submitter import AsyncExecutor
from .config import LoggingConfig
from .logger_factory import (
    LoggerFactory,
    LogType,
    MetricLogger,
    PerformanceLogger,
    SystemLogger,
)


ROOT_LOGGER = {
    "system": SystemLogger,
    "performance": PerformanceLogger,
    "metric": MetricLogger,
}


class LoggerManager(AsyncExecutor, LoggerFactory):

    """
    Initialize loggers for Pipeline

    :param config: Path to yaml or stringified yaml
    """

    def __init__(self, config: str = ""):
        self.config = LoggingConfig.from_config(config).dict()
        super().__init__(config=self.config)

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
            logger = self.logger.get(LogType[log_type])
            if logger:
                logger.log(value=value, tag=tag, *args, **kwargs)

    def callback(self, future: Future):
        exception = future.exception()
        if exception is not None:
            self.system.log(
                f"Exception occurred during async logging job: {repr(exception)}",
                level="error",
            )

    @property
    def system(self) -> Optional[SystemLogger]:
        return self.logger[LogType.SYSTEM]

    @property
    def performance(self) -> Optional[PerformanceLogger]:
        return self.logger[LogType.PERFORMANCE]

    @property
    def metric(self) -> Optional[MetricLogger]:
        return self.logger[LogType.METRIC]
