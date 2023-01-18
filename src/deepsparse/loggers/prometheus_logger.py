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


try:
    from prometheus_client import (
        REGISTRY,
        Summary,
        start_http_server,
        write_to_textfile,
    )

    prometheus_import_error = None
except Exception as prometheus_import_err:
    REGISTRY = None
    Summary = None
    start_http_server = None
    write_to_textfile = None
    prometheus_import_error = prometheus_import_err


__all__ = ["PrometheusLogger"]

_LOGGER = logging.getLogger(__name__)
SUPPORTED_DATA_TYPES = (int, float)


class PrometheusLogger(BaseLogger):
    """
    DeepSparse logger that continuously exposes the collected logs over the
    Prometheus python client at the specified port.

    :param port: the port used by the client. Default is 6100
    :param text_log_save_frequency: the frequency of saving the text log
        files. E.g. if `text_log_save_frequency` = 10, text logs are
        exported after every tenth forward pass. Default set to 10
    :param text_log_save_dir: the directory where the text log files
        are saved. By default, the python working directory
    :param text_log_file_name: the name of the text log file.
        Default: `prometheus_logs.prom`
    """

    def __init__(
        self,
        port: int = 6100,
        text_log_save_frequency: int = 10,
        text_log_save_dir: str = os.getcwd(),
        text_log_file_name: Optional[str] = None,
    ):
        _check_prometheus_import()

        self.port = port
        self.text_log_save_frequency = text_log_save_frequency
        self.text_log_save_dir = text_log_save_dir
        self.text_log_file_path = os.path.join(
            text_log_save_dir, text_log_file_name or "prometheus_logs.prom"
        )
        self._prometheus_metrics = defaultdict(str)

        self._setup_client()
        self._counter = 0

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        Collect information from the pipeline and pipe it them to the stdout

        :param identifier: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """

        # needs to adhere to
        # https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
        formatted_identifier = (
            identifier.replace(".", "__").replace("-", "__").replace("/", "__")
        )
        prometheus_metric = self._prometheus_metrics.get(formatted_identifier)
        if prometheus_metric is None:
            prometheus_metric = self._add_metric_to_registry(
                formatted_identifier, category
            )
        prometheus_metric.observe(self._validate(value))
        self._export_metrics_to_textfile()

    def __str__(self):
        logger_info = f"  port: {self.port}"
        return f"{self.__class__.__name__}:\n{logger_info}"

    def _export_metrics_to_textfile(self):
        # export the metrics to a text file with
        # the specified frequency
        os.makedirs(self.text_log_save_dir, exist_ok=True)
        if self._counter % self.text_log_save_frequency == 0:
            write_to_textfile(self.text_log_file_path, REGISTRY)
            self._counter = 0
        self._counter += 1

    def _add_metric_to_registry(self, identifier: str, category: str) -> Summary:
        description = (
            """Summary metric for identifier: {identifier} | Category: {category}"""
        )
        prometheus_metric = Summary(
            identifier,
            description.format(identifier=identifier, category=category),
            registry=REGISTRY,
        )
        self._prometheus_metrics[identifier] = prometheus_metric
        return prometheus_metric

    def _setup_client(self):
        # starts the Prometheus client
        start_http_server(port=self.port)
        _LOGGER.info(f"Prometheus client: started. Using port: {self.port}.")

    def _validate(self, value: Any) -> Any:
        # make sure we are passing a value that is
        # a valid metric by prometheus client's standards
        if not isinstance(value, SUPPORTED_DATA_TYPES):
            raise ValueError(
                "Prometheus logger expects the incoming values "
                f"to be one of the type: {SUPPORTED_DATA_TYPES}"
            )
        return value


def _check_prometheus_import():
    if prometheus_import_error is not None:
        _LOGGER.error(
            "Attempting to instantiate a PrometheusLogger object but unable to import "
            "prometheus. Check that prometheus requirements have been installed"
        )
        raise prometheus_import_error
