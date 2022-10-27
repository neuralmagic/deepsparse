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
"""
from typing import Any

from deepsparse.loggers import BaseLogger, MetricCategories


__all__ = ["PrometheusLogger"]


class PrometheusLogger(BaseLogger):
    """ """

    def __init__(self, port):
        self.port = port

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        Collect information from the pipeline and pipe it them to the stdout

        :param identifier: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        raise NotImplementedError()
