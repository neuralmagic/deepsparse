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
import logging
from typing import Any

from deepsparse.loggers import BaseLogger, MetricCategories


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")

__all__ = ["PythonLogger"]


class PythonLogger(BaseLogger):
    """
    Generic BaseLogger abstract class meant to define interfaces
    for the loggers that support various monitoring services APIs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        Collect information from the pipeline and print them to the console

        :param identifier: The identifier of the log
             By convention should have the following structure:
             {pipeline_name}.{target_name}.{optional_identifier_1}.{optional_identifier_2}.{...}
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        msg = (
            f"Identifier: {identifier} | Category: {category.value} "
            f"| Logged Data Type: {type(value)}"
        )
        logging.info(msg=msg)
