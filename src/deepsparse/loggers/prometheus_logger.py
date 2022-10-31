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
Implementation of the Prometheus Logger
"""
import logging
import os
from collections import defaultdict
from typing import Any, Optional

from deepsparse.loggers import BaseLogger, MetricCategories
from prometheus_client import REGISTRY, Summary, start_http_server, write_to_textfile


__all__ = ["PrometheusLogger"]

_LOGGER = logging.getLogger(__name__)


class PrometheusLogger(BaseLogger):
    """
    DeepSparse logger that continuously exposes the collected logs over the
    Prometheus python client at the specified port.
    """

    def __init__(
        self,
        port: int = 6100,
        text_log_save_dir_frequency: int = 50,
        text_log_save_dir: str = os.getcwd(),
        text_log_file_name: Optional[str] = None,
    ):

        self.port = port
        # until we have done research into persistent logs, lets save logs as .txt files
        self.text_log_save_dir_frequency = text_log_save_dir_frequency
        self.text_log_file_name = os.path.join(
            text_log_save_dir, text_log_file_name or "prometheus_logger_logs.prom"
        )
        self.prometheus_metrics = defaultdict(str)

        self._setup_client()
        self._counter = 0

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        Collect information from the pipeline and pipe it them to the stdout

        :param identifier: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """

        value = self._validate(value)
        if value is None:
            return
        formatted_identifier = identifier.replace(".", ":")
        prometheus_metric = self.prometheus_metrics.get(formatted_identifier)
        if prometheus_metric is None:
            prometheus_metric = self._add_metric_to_registry(
                formatted_identifier, category
            )
        prometheus_metric.observe(value)
        self._conditionally_export_metrics_to_textfile()

    def _add_metric_to_registry(self, identifier: str, category: str) -> Summary:
        description = (
            """Summary metric for identifier: {identifier} | Category: {category}"""
        )
        prometheus_metric = Summary(
            identifier,
            description.format(identifier=identifier, category=category),
            registry=REGISTRY,
        )
        self.prometheus_metrics[identifier] = prometheus_metric
        return prometheus_metric

    def _conditionally_export_metrics_to_textfile(self):
        if self._counter % self.text_log_save_dir_frequency == 0:
            text_log_file_name = os.path.join(
                self.text_log_save_dir, self.text_log_file_name
            )
            write_to_textfile(text_log_file_name, REGISTRY)
            self._counter = 0
        self._counter += 1

    def _setup_client(self):
        """
        Starts the Prometheus client
        """
        start_http_server(port=self.port)
        _LOGGER.info(f"Prometheus client: started. Using port: {self.port}.")

    def _validate(self, value):
        from pydantic import BaseModel

        if isinstance(value, BaseModel) or isinstance(value, list):
            return None
        return value
