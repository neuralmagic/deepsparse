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
import textwrap
from typing import Any, Callable

from deepsparse.loggers import BaseLogger, MetricCategories
from deepsparse.loggers.helpers import NO_MATCH, finalize_identifier, match_and_extract


__all__ = ["FunctionLogger"]


class FunctionLogger(BaseLogger):
    """
    DeepSparse logger that applies a metric function to
    a raw value (received in the log() method) and then
    passes it to a child logger.

    :param logger: A child DeepSparse Logger object
    :param target_identifier: Needs to match the
        `identifier` (argument to the log() method).
        Only then the FunctionLogger applies
        the metric function and logs the data
    :param function_name: Name of the metric function
    :param function: The metric function to be applied
    :param frequency: The frequency with which the metric
        name is to be applied
    """

    def __init__(
        self,
        logger: BaseLogger,
        target_identifier: str,
        function: Callable[[Any], Any],
        function_name: str = None,
        frequency: int = 1,
    ):

        self.logger = logger
        self.target_identifier = target_identifier
        self.function = function
        self.function_name = function_name or function.__name__
        self.frequency = frequency

        self._function_call_counter = 0

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        If the identifier matches the target identifier, the value of interest
        is being extracted and the metric function is applied to the extracted value.
        The result is then logged to the child logger with the appropriate
        frequency

        :param identifier: The name of the item that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        extracted_value, remainder = match_and_extract(
            template=self.target_identifier,
            identifier=identifier,
            value=value,
            category=category,
        )
        # `None` can be a valid, extracted value.
        #  Hence, we check for `NO_MATCH` flag
        if extracted_value != NO_MATCH:
            if self._function_call_counter % self.frequency == 0:
                mapped_value = self.function(extracted_value)
                self.logger.log(
                    identifier=finalize_identifier(
                        identifier, category, self.function_name, remainder
                    ),
                    value=mapped_value,
                    category=category,
                )
                self._function_call_counter = 0
            self._function_call_counter += 1

    def __str__(self):
        def _indent(value):
            return textwrap.indent(str(value), prefix="  ")

        def _shorten(text, max_num_lines=10, placeholder="[...]"):
            lines = text.split("\n")
            if len(lines) <= max_num_lines:
                return text
            lines_to_keep = lines[:max_num_lines]
            lines_to_keep.append(placeholder)
            return "\n".join(lines_to_keep)

        function_info = (
            f"target identifier: {self.target_identifier}\n"
            f"function name: {self.function.__name__}\n"
            f"frequency: {self.frequency}\n"
            f"target_logger:\n{_indent(self.logger)}"
        )
        return f"{self.__class__.__name__}:\n{_indent(_shorten(function_info))}"
