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

from deepsparse.loggers.constants import SystemGroups
from deepsparse.loggers.registry.loggers.base_logger import BaseLogger


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
_tag_TO_METRIC_TYPE = {
    "prediction_latency": Histogram,
    SystemGroups.RESOURCE_UTILIZATION: Gauge,
    f"{SystemGroups.REQUEST_DETAILS}/successful_request": Counter,
    f"{SystemGroups.REQUEST_DETAILS}/input_batch_size": Histogram,
}
_SUPPORTED_DATA_TYPES = (int, float)
_DESCRIPTION = "{metric_name} metric for tag: {tag} | log_type: {log_type}"


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
        port: int = 9090,
        text_log_save_dir: str = os.getcwd(),
        text_log_file_name: Optional[str] = None,
        **_ignore_args,
    ):
        _check_prometheus_import()

        self.port = port
        self.text_log_save_dir = text_log_save_dir
        self.text_log_file_path = os.path.join(
            text_log_save_dir, text_log_file_name or "prometheus_logs.prom"
        )
        self._prometheus_metrics = defaultdict(str)

        self._setup_client()

    def log(
        self,
        tag: str,
        value: Any,
        log_type: str,
        capture: Optional[str] = None,
        **kwargs,
    ):
        """
        Collect information from the pipeline and pipe it them to the stdout

        :param tag: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param log_type: The metric log_type that the log belongs to
        :param kwargs: Additional keyword arguments to pass to the logger
        """

        pipeline_name = tag
        prometheus_metric = self._get_prometheus_metric(
            capture or tag, log_type, **kwargs
        )
        if prometheus_metric is None:
            warnings.warn(
                f"The tag {tag} cannot be matched with any "
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
        tag: str,
        log_type: str,
        **kwargs,
    ) -> Optional[_PrometheusMetric]:
        saved_metric = self._prometheus_metrics.get(tag)
        if saved_metric is None:
            return self._add_metric_to_registry(tag, log_type, **kwargs)
        return saved_metric

    def _add_metric_to_registry(
        self,
        tag: str,
        log_type: str,
        **kwargs,
    ) -> Optional[_PrometheusMetric]:
        prometheus_metric = get_prometheus_metric(tag, log_type, REGISTRY, **kwargs)
        self._prometheus_metrics[tag] = prometheus_metric
        return prometheus_metric

    def __str__(self):
        logger_info = f"  port: {self.port}"
        return f"{self.__class__.__name__}:\n{logger_info}"

    def _export_metrics_to_textfile(self):
        # export the metrics to a text file with
        # the specified frequency
        os.makedirs(self.text_log_save_dir, exist_ok=True)
        write_to_textfile(self.text_log_file_path, REGISTRY)

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
    tag: str, log_type: str, registry: CollectorRegistry, **kwargs
) -> Optional["MetricWrapperBase"]:  # noqa: F821
    """
    Get a Prometheus metric object for the given tag and log_type.

    :param tag: The name of the thing that is being logged.
    :param log_type: The metric log_type that the log belongs to
    :param registry: The Prometheus registry to which the metric should be added
    :return: The Prometheus metric object or None if the tag not supported
    """

    if log_type == "system":
        metric = _get_metric_from_the_mapping(tag)
    else:
        metric = Summary

    if metric is None:
        return None

    pipeline_name = tag
    return metric(
        name=format_tag(tag),
        documentation=_DESCRIPTION.format(
            metric_name=metric._type, tag=tag, log_type=log_type
        ),
        labelnames=["pipeline_name"] if pipeline_name else [],
        registry=registry,
    )


def _get_metric_from_the_mapping(
    tag: str, metric_type_mapping: Dict[str, str] = _tag_TO_METRIC_TYPE
) -> Optional[Type["MetricWrapperBase"]]:  # noqa: F821
    for system_group_name, metric_type in metric_type_mapping.items():
        """
        Attempts to get the metric type given the tag and system_group_name.
        There are two cases:
        Case 1) If system_group_name contains both the group name and the tag,
            e.g. "request_details/successful_request", the match requires the tag
            to end with the system_group_name,
            e.g. "pipeline_name/request_details/successful_request".
        Case 2) If system_group_name contains only the group name,
            e.g. "prediction_latency",
            the match requires the system_group_name to be
            contained within the tag
            e.g. prediction_latency/pipeline_inputs
        """
        if ("/" in system_group_name and tag.endswith(system_group_name)) or (
            system_group_name in tag
        ):
            return metric_type


def format_tag(tag: str, namespace: str = _NAMESPACE) -> str:
    """
    Replace forbidden characters with `__` so that the tag
    digested by prometheus adheres to
    https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
    :param tag: The tag to be formatted
    :return: The formatted tag
    """
    return f"{namespace}_{re.sub(r'[^a-zA-Z0-9_]+', '__', tag).lower()}"


def _check_prometheus_import():
    if prometheus_import_error is not None:
        _LOGGER.error(
            "Attempting to instantiate a PrometheusLogger object but unable to import "
            "prometheus. Check that prometheus requirements have been installed"
        )
        raise prometheus_import_error
