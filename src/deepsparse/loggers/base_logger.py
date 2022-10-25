"""
Base implementation of the logger
"""
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
from typing import Any, Optional


class BaseLogger(ABC):
    """
    Generic BaseLogger abstract class meant to define interfaces
    for the loggers that support various monitoring services APIs.
    """

    @abstractmethod
    def log(
        self,
        pipeline_name: str,
        target: str,
        value: Any,
        category: Optional[str] = None,
    ):
        """
        The main method to collect information from the pipeline
        and then possibly process the information and pass it to
        the monitoring service

        :param pipeline_name: The name of the pipeline that the log relates to
        :param target: The identifier of the log
            The target may be a single string:
                e.g. target = "pipeline_input"
            Or, if identifier is complex, multiple strings are to be concatenated
            using dot as a separator:
                e.g. target = "pipeline_input.embedding"
        :param value: The data structure that is logged
        :param category: The metric category that the log belongs to
        """
        raise NotImplementedError()
