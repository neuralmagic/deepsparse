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
from concurrent.futures import Future
from typing import Any

from deepsparse.loggers.async_executor import AsyncExecutor
from deepsparse.loggers.config import LoggingConfig
from deepsparse.loggers.logger_factory import (
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
    Initialize loggers for Pipeline and create entrypoints to log

    Lifecycle of instantiation:
        1. Pydantic validation/parser
        2. In LoggerFactory, instantiate leaf logger as singleton and
         use them to instantiate for system, performance
         and metric root loggers
        3. In root logger instantiation, for each tag, func, freq,
         generate a default dict to organize the params from the config to
         facilliate filter rule matching (by tag, by freq, by capture)

    Entrypoints:
        * .log          -> log async to the root logger, selected by log_type
        * .system       -> log async to the system logger
        * .performance  -> log async to the performance logger
        * .metric       -> log async to the metric logger

    Note:
        * To access what leaf loggers are being used
         .root_logger_factory["system" or "performance" or "metric"]
        * To access the frequency filter counters
         .root_logger_factory[...].counter

    :param config: Path to yaml or stringified yaml

    """

    def __init__(self, config: str = ""):
        self.config = LoggingConfig.from_config(config).model_dump()
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

    def run(self, value: Any, tag: str, log_type: str, *args, **kwargs):
        log_type = log_type.upper()
        if log_type in LogType.__members__:
            logger = self.logger.get(LogType[log_type])
            if logger:
                logger.log(value=value, tag=tag, *args, **kwargs)

    def callback(self, future: Future):
        exception = future.exception()
        if exception is not None:
            logging.error(
                value=f"Exception occurred during async logging job: {repr(exception)}",
            )

    def system(self, *args, **kwargs):
        self.log(
            log_type="system",
            *args,
            **kwargs,
        )

    def performance(self, *args, **kwargs):
        self.log(
            log_type="performance",
            *args,
            **kwargs,
        )

    def metric(self, *args, **kwargs):
        self.log(
            log_type="metric",
            *args,
            **kwargs,
        )
