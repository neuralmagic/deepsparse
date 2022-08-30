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

"""
A manager that oversees all the pipeline loggers
"""

from typing import Any, Dict, List, Union

from deepsparse.loggers import BaseLogger
from deepsparse.loggers.prometheus_logger import PrometheusLogger
from deepsparse.timing.timing_schema import InferenceTimingSchema


__all__ = ["ManagerLogger"]

SUPPORTED_LOGGERS = [PrometheusLogger]


class ManagerLogger(BaseLogger):
    """
    Object that contains multiple loggers for
    the given inference pipeline.

    The envisioned lifecycle of a manager logger:

    ```
    pipeline = ... # define a pipeline
    pipeline_name = pipeline.name # fetch the pipeline name

    # create a ManagerLogger
    logger_manager = ManagerLogger("logger_name")
    or
    logger_manager = ManagerLogger(["logger_name_1", "logger_name_2", ...])

    # log the data for the particular inference pipeline
    timings, data = ...
    logger.log_latency(pipeline_name, timings)
    logger.log_data(pipeline_name, data)

    ```
    :param loggers: Logger class instances that will be contained
        within the scope of a manager. This can be either a single logger
        or a  list of loggers (for multiple loggers).
    """

    def __init__(
        self,
        loggers: Union[BaseLogger, List[BaseLogger]],
    ):
        self._supported_loggers = SUPPORTED_LOGGERS
        self._loggers = self._validate(loggers)

    @property
    def identifier(self) -> List[str]:
        """
        :return: a list of identifiers for every logger
            that is being contained in scope of the
            manager logger
        """
        return [identifier for identifier in self.loggers]

    @property
    def loggers(self) -> Dict[str, BaseLogger]:
        """
        :return: The mapping from the logger identifier
        to the BaseLogger instance
        """
        return self._loggers

    def log_latency(self, pipeline_name: str, inference_timing: InferenceTimingSchema):
        """
        Logs the inference latency information to the all the loggers in manager's scope

        :param pipeline_name: The name of the inference pipeline from which the
            logger consumes the inference information to be monitored
        :param inference_timing: Pydantic model that contains information
            about time deltas of various processes within the inference pipeline
        """
        for name, logger in self.loggers.items():
            logger.log_latency(pipeline_name, inference_timing)

    def log_data(self, pipeline_name: str, inputs: Any, outputs: Any):
        """
        Logs the inference inputs and outputs to all the loggers in manager's scope

        :param pipeline_name: The name of the inference pipeline from which the
            logger consumes the data information to be monitored
        :param inputs: the data received and consumed by the inference
            pipeline
        :param outputs: the data returned by the inference pipeline
        """
        for name, logger in self.loggers.items():
            logger.log_data(pipeline_name, inputs, outputs)

    def _validate(
        self, loggers: Union[BaseLogger, List[BaseLogger]]
    ) -> Dict[str, BaseLogger]:
        if not isinstance(loggers, List):
            loggers = [loggers]
        _loggers = {}
        for logger in loggers:
            is_logger_supported = any(
                isinstance(logger, supported_logger)
                for supported_logger in self._supported_loggers
            )
            if not is_logger_supported:
                raise ValueError(
                    f"Attempting to create an unknown "
                    f"pipeline logger: {logger.identifier} logger"
                )
            else:
                _loggers[logger.identifier] = logger

        return _loggers
