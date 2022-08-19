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
from typing import Any


__all__ = ["PipelineLogger"]


class PipelineLogger(ABC):
    """
    Generic PipelineLogger abstract class meant to define interfaces
    for the loggers that support various monitoring services APIs.
    """

    def __init__(self):
        pass

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
