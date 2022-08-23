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

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Dict
from typing import Iterable as iterable_type
from typing import Optional, Union

from deepsparse.pipeline import Pipeline


__all__ = ["PipelineLogger", "LoggerManager"]


class PipelineLogger(ABC):
    """
    Generic PipelineLogger abstract class meant to define interfaces
    for the loggers that support various monitoring services APIs.

    :param pipeline_name: The name of the inference pipeline from which the
        logger consumes the inference information to be monitored
    :param identifier: The name of the monitoring service that the
        PipelineLogger uses to log the inference data
    """

    def __init__(self, identifier: str, pipeline_name: Optional[str] = None):
        self._pipeline_name = pipeline_name
        self._identifier = identifier

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def pipeline_name(self) -> str:
        return self._pipeline_name

    @pipeline_name.setter
    def pipeline_name(self, value: str):
        """
        Set the name of the inference pipeline that this
        logger collects the inference data from

        :param value: pipeline name to be related with
            this logger
        """
        if self._pipeline_name is not None:
            raise ValueError(
                "Attempting to set the pipeline name for the pipeline logger, "
                "but this logger is already associated with the existing pipeline: "
                f"{self._pipeline_name}"
            )
        else:
            self._pipeline_name = value

    def __str__(self):
        return (
            f"Logger for the pipeline: {self.pipeline_name}; "
            f"using monitoring service: {self.identifier}"
        )

    @abstractmethod
    def log_latency(
        self, inference_timing: "InferenceTimingSchema", **kwargs  # noqa F821
    ):
        """
        Logs the inference latency information to the appropriate monitoring service
        :param inference_timing: pydantic model that holds the information about
            the inference latency of a forward pass
        """
        raise NotImplementedError()

    @abstractmethod
    def log_data(self, inputs: Any, outputs: Any):
        """
        Logs the inference inputs and outputs to the appropriate monitoring service
        :param inputs: the data received and consumed by the inference
            pipeline
        :param outputs: the data returned by the inference pipeline
        """
        raise NotImplementedError()


class LoggerManager:
    """
    Object that contains multiple loggers for
    the given inference pipeline.

    Below, the envisioned lifecycle of a logger
    manager:

    ```
    pipeline = ... # define a pipeline

    # create a LoggerManager
    logger_manager = LoggerManager.from_pipeline(pipeline)
    # define the set of loggers
    logger = LoggerA() or logger = [LoggerA(), LoggerB(), ...]
    # add loggers to the LoggerManager
    logger_manager.add(logger)
    ...
    ```
    :param pipeline_name: The name of the pipeline the
        logger manager is associated with
    """

    def __init__(self, pipeline_name: str):
        self._pipeline_name = pipeline_name
        self._loggers = {}

    @property
    def pipeline_name(self) -> str:
        """
        :return: The name of the pipeline the
        logger manager is associated with
        """
        return self._pipeline_name

    @property
    def loggers(self) -> Dict[str, PipelineLogger]:
        """
        :return: The mapping from the logger name (identifier)
        to the PipelineLogger instance
        """
        return self._loggers

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> "LoggerManager":
        """
        Factory method to create LoggerManager instance associated
        with an existing pipeline

        :param pipeline: Pipeline class object
        :return: LoggerManager class object
        """
        return cls(pipeline_name=pipeline.task)  # TODO: Something better than task?

    def add(
        self, pipeline_logger: Union[PipelineLogger, iterable_type[PipelineLogger]]
    ):
        """
        Adds one or multiple pipeline loggers to the manager

        :param pipeline_logger: An instance of a pipeline logger or an iterable
            that holds multiple pipeline loggers
        """
        if not isinstance(pipeline_logger, Iterable):
            pipeline_logger = [pipeline_logger]

        for logger in pipeline_logger:
            if logger.pipeline_name:
                raise ValueError(
                    f"Expected to add a logger that is yet "
                    f"not assigned to any pipeline."
                    f"However, logger: {logger} is already assigned to pipeline: "
                    f"{logger.pipeline_name}"
                )
            logger.pipeline_name = self.pipeline_name
            self._loggers[logger.identifier] = logger
