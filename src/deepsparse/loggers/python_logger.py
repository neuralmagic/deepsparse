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
Implementation of the Python Logger that logs to the stdout
"""
from datetime import datetime
from typing import Any

from deepsparse.loggers import BaseLogger, MetricCategories


__all__ = ["PythonLogger"]


class PythonLogger(BaseLogger):
    """
    Python logger that writes the collected logs to stdout
    """

    def __init__(self, **kwargs):

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        Collect information from the pipeline and pipe it them to the stdout

        :param identifier: The identifier of the log
             By default should consist of at least one string (name of the pipeline):
                e.g. identifier = "pipeline_name"
            If identifier is to be more complex, optional strings are to be concatenated
            using dot as a separator:
                e.g. identifier = "pipeline_name.some_argument_1.some_argument_2"
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        msg = (
            f" Identifier: {identifier} | Category: {category.value} "
            f"| Logged Data: {value}"
        )
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + msg)
