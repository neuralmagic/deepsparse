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
Implementation of the Function Logger
"""
from typing import Any, Callable

from deepsparse.loggers import BaseLogger, MetricCategories
from deepsparse.loggers.helpers import possibly_extract_value, match


__all__ = ["FunctionLogger"]


class FunctionLogger(BaseLogger):
    """
    DeepSparse logger that applies functions to raw log values
    according to the specified config file

    :param logger: A DeepSparse Logger object
    """

    def __init__(
        self,
        logger: BaseLogger,
        identifier: str,
        function_name: Any,
        function: Callable[[Any], Any],
        frequency: int):

        self.logger = logger
        self.identifier = identifier
        self.function_name = function_name
        self.frequency = frequency
        self.function = function

        self._counts = 0

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        :param identifier: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        is_match, sub_identifier = match(template=self.identifier, identifier=identifier)
        if is_match:
            if self._counts %  self.frequency == 0:
                filtered_value = possibly_extract_value(value, sub_identifier)
                mapped_value = self.function(filtered_value)
                self.logger.log(identifier=identifier, value=mapped_value, category=category)
                self._counts = 0
            self._counts += 1


