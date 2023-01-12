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
Implementation of the Multi Logger that serves as a
container for holding multiple loggers
"""
import textwrap
from typing import Any, List

from deepsparse.loggers import BaseLogger, MetricCategories


__all__ = ["MultiLogger"]


class MultiLogger(BaseLogger):
    """
    A logger that holds a list of loggers and logs to all of them.
    """

    def __init__(self, loggers: List[BaseLogger]):
        self.loggers = loggers

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """

        :param identifier: The name of the item that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        for logger in self.loggers:
            logger.log(identifier, value, category)

    def __str__(self):
        text = "\n".join([str(logger) for logger in self.loggers])
        return f"{self.__class__.__name__}:\n{textwrap.indent(text, prefix='  ')}"
