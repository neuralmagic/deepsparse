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

from typing import Any, Dict, List, Set, Union

from deepsparse import PipelineLogger
from deepsparse.pipeline_loggers.prometheus_pipeline_logger import (
    IDENTIFIER as PROMETHEUS_IDENTIFIER,
)
from deepsparse.pipeline_loggers.prometheus_pipeline_logger import PrometheusLogger
from deepsparse.timing.timing_schema import InferenceTimingSchema


__all__ = ["LoggerManager"]

SUPPORTED_LOGGERS = {PROMETHEUS_IDENTIFIER}


class LoggerManager:
    """
    Object that contains multiple loggers for
    the given inference pipeline.

    The envisioned lifecycle of a logger
    manager:

    ```
    pipeline = ... # define a pipeline
    pipeline_name = pipeline.name # fetch the pipeline name

    # create a LoggerManager
    logger_manager = LoggerManager("logger_name")
    or
    logger_manager = LoggerManager(["logger_name_1", "logger_name_2", ...])

    # log the data for the particular inference pipeline

    timings, data = ...
    logger.log_timings(pipeline_name, timings)
    logger.log_data(pipeline_name, data)

    ```

    :param logger_identifiers: The identifier(s) of the logger types that
        will be created within the scope of the manager. This can be either a
        single identifier, or a list of identifiers (for multiple loggers).
    :param supported_loggers: A set of supported logger identifiers; listed in
        `SUPPORTED_LOGGERS` variable
    """

    def __init__(
        self,
        logger_identifiers: Union[str, List[str]],
        supported_loggers: Set[str] = SUPPORTED_LOGGERS,
    ):
        self._supported_loggers = supported_loggers
        self._loggers = self._setup_loggers(logger_identifiers)

    @property
    def loggers(self) -> Dict[str, PipelineLogger]:
        """
        :return: The mapping from the logger identifier
        to the PipelineLogger instance
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

    def _setup_loggers(
        self, logger_identifiers: Union[str, List[str]]
    ) -> Dict[str, PipelineLogger]:
        _loggers = {}
        if isinstance(logger_identifiers, str):
            logger_identifiers = [logger_identifiers]

        for logger_identifier in logger_identifiers:
            # the if-statement below shall be expanded along with
            # the new PipelineLogger implementations.
            # Essentially a switch
            # statement for instantiating PipelineLoggers
            if logger_identifier not in self._supported_loggers:
                raise ValueError(
                    "Attempting to create a pipeline logger with an "
                    f"unknown identifier: {logger_identifier}. Supported "
                    f"identifiers are: {self._supported_loggers}"
                )

            elif logger_identifier == PROMETHEUS_IDENTIFIER:
                _loggers[logger_identifier] = PrometheusLogger()

            else:
                raise NotImplementedError()

        return _loggers
