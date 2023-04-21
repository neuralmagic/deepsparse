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
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, Optional, Type, Union

from deepsparse.loggers import BaseLogger, MetricCategories, SystemGroups
from deepsparse.loggers.helpers import unwrap_logged_value


try:
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
        write_to_textfile,
    )

    prometheus_import_error = None
except Exception as prometheus_import_err:
    REGISTRY = None
    Summary = None
    Histogram = None
    Gauge = None
    Counter = None
    CollectorRegistry = None
    start_http_server = None
    write_to_textfile = None
    prometheus_import_error = prometheus_import_err


__all__ = ["PrometheusLogger"]

_LOGGER = logging.getLogger(__name__)

_NAMESPACE = "deepsparse"
_PrometheusMetric = Union[Histogram, Gauge, Summary, Counter]
_IDENTIFIER_TO_METRIC_TYPE = {
    "prediction_latency": Histogram,
    SystemGroups.RESOURCE_UTILIZATION: Gauge,
    f"{SystemGroups.REQUEST_DETAILS}/successful_request": Counter,
    f"{SystemGroups.REQUEST_DETAILS}/input_batch_size": Histogram,
}
_SUPPORTED_DATA_TYPES = (int, float)
_DESCRIPTION = (
    "{metric_name} metric for identifier: {identifier} | Category: {category}"
)


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

    def log(self, identifier: str, value: Any, category: MetricCategories, **kwargs):
        """
        Collect information from the pipeline and pipe it them to the stdout

        :param identifier: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        :param kwargs: Additional keyword arguments to pass to the logger
        """

        pipeline_name = kwargs.get("pipeline_name")
        for identifier, value in unwrap_logged_value(value, identifier):
            prometheus_metric = self._get_prometheus_metric(
                identifier, category, **kwargs
            )
            if prometheus_metric is None:
                warnings.warn(
                    f"The identifier {identifier} cannot be matched with any "
                    f"of the Prometheus metrics and will be ignored."
                )
                return
            if pipeline_name:
                prometheus_metric.labels(pipeline_name=pipeline_name).observe(
                    self._validate(value)
                )
            else:
                prometheus_metric.observe(self._validate(value))
        self._export_metrics_to_textfile()

    def _get_prometheus_metric(
        self,
        identifier: str,
        category: MetricCategories,
        **kwargs,
    ) -> Optional[_PrometheusMetric]:
        saved_metric = self._prometheus_metrics.get(identifier)
        if saved_metric is None:
            return self._add_metric_to_registry(identifier, category, **kwargs)
        return saved_metric

    def _add_metric_to_registry(
        self,
        identifier: str,
        category: str,
        **kwargs,
    ) -> Optional[_PrometheusMetric]:
        # add a new metric to the registry
        prometheus_metric = get_prometheus_metric(
            identifier, category, REGISTRY, **kwargs
        )
        self._prometheus_metrics[identifier] = prometheus_metric
        return prometheus_metric

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

    def _setup_client(self):
        # starts the Prometheus client
        start_http_server(port=self.port)
        _LOGGER.info(f"Prometheus client: started. Using port: {self.port}.")

    def _validate(self, value: Any) -> Any:
        # make sure we are passing a value that is
        # a valid metric by prometheus client's standards
        if not isinstance(value, _SUPPORTED_DATA_TYPES):
            raise ValueError(
                "Prometheus logger expects the incoming values "
                f"to be one of the type: {_SUPPORTED_DATA_TYPES}, "
                f"but received: {type(value)}"
            )
        return value


def get_prometheus_metric(
    identifier: str, category: MetricCategories, registry: CollectorRegistry, **kwargs
) -> Optional["MetricWrapperBase"]:  # noqa: F821
    """
    Get a Prometheus metric object for the given identifier and category.

    :param identifier: The name of the thing that is being logged.
    :param category: The metric category that the log belongs to
    :param registry: The Prometheus registry to which the metric should be added
    :return: The Prometheus metric object or None if the identifier not supported
    """
    if category == MetricCategories.SYSTEM:
        metric = _get_metric_from_the_mapping(identifier)
    else:
        metric = Summary

    if metric is None:
        return None

    pipeline_name = kwargs.get("pipeline_name")
    return metric(
        name=format_identifier(identifier),
        documentation=_DESCRIPTION.format(
            metric_name=metric._type, identifier=identifier, category=category
        ),
        labelnames=["pipeline_name"] if pipeline_name else [],
        registry=registry,
    )


def _get_metric_from_the_mapping(
    identifier: str, metric_type_mapping: Dict[str, str] = _IDENTIFIER_TO_METRIC_TYPE
) -> Optional[Type["MetricWrapperBase"]]:  # noqa: F821
    for system_group_name, metric_type in metric_type_mapping.items():
        """
        Attempts to get the metric type given the identifier and system_group_name.
        There are two cases:
        Case 1) If system_group_name contains both the group name and the identifier,
            e.g. "request_details/successful_request", the match requires the identifier
            to end with the system_group_name,
            e.g. "pipeline_name/request_details/successful_request".
        Case 2) If system_group_name contains only the group name,
            e.g. "prediction_latency",
            the match requires the system_group_name to be
            contained within the identifier
            e.g. prediction_latency/pipeline_inputs
        """
        if ("/" in system_group_name and identifier.endswith(system_group_name)) or (
            system_group_name in identifier
        ):
            return metric_type


def format_identifier(identifier: str, namespace: str = _NAMESPACE) -> str:
    """
    Replace forbidden characters with `__` so that the identifier
    digested by prometheus adheres to
    https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
    :param identifier: The identifier to be formatted
    :return: The formatted identifier
    """
    return f"{namespace}_{re.sub(r'[^a-zA-Z0-9_]', '__', identifier).lower()}"


def _check_prometheus_import():
    if prometheus_import_error is not None:
        _LOGGER.error(
            "Attempting to instantiate a PrometheusLogger object but unable to import "
            "prometheus. Check that prometheus requirements have been installed"
        )
        raise prometheus_import_error
