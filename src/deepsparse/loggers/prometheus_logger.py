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
import logging
import os
from collections import defaultdict
from typing import Any, Optional

from deepsparse.loggers.base_logger import BaseLogger
from prometheus_client import REGISTRY, Histogram, start_http_server, write_to_textfile


__all__ = ["PrometheusLogger"]

_LOGGER = logging.getLogger(__name__)


class PrometheusLogger(BaseLogger):
    """
    Logger that uses the official Prometheus python client
    (https://github.com/prometheus/client_python) to monitor the
    inference pipeline

    :param port: the port used by the client. Default is 8000
    :param text_log_save_dir: the directory where the text log files
        are saved. By default, the python working directory
    :param text_log_save_freq: the frequency of saving the text log
        files. E.g. if `text_log_save_freq` = 10, text logs are dumped
        after every tenth forward pass (the number of forward passes
        equals the number of times `log_latency` method is called).
    :param text_log_file_name: the name of the text log file.
        Default: `prometheus_logs.prom`.
    """

    def __init__(
        self,
        port: int = 6100,
        text_log_save_dir: str = os.getcwd(),
        text_log_save_freq: int = 10,
        text_log_file_name: Optional[str] = None,
    ):
        self.port = port
        self.text_log_save_freq = text_log_save_freq
        self.text_log_save_dir = text_log_save_dir
        self.text_log_file_name = text_log_file_name or f"{self.identifier}_logs.prom"
        self._setup_client()

        # the data structure responsible for the instrumentation
        # of the metrics
        self.metrics = defaultdict(lambda: defaultdict(str))
        # internal counter tracking the number of calls per each
        # pipeline i.e. self._counter: Dict[str, int]
        self._counter = {}

        super().__init__()

    def __str__(self):
        return (
            f"{self.__class__.__name__}(port={self.port}, text_log_save_dir="
            f"{self.text_log_save_dir}, text_log_save_freq={self.text_log_save_freq}, "
            f"text_log_file_name={self.text_log_file_name})"
        )

    @property
    def identifier(self) -> str:
        return "prometheus"

    @property
    def text_logs_path(self) -> str:
        """
        :return: The path to the text file with logs
        """
        return os.path.join(self.text_log_save_dir, self.text_log_file_name)

    @property
    def counter(self) -> int:
        """
        While self._counter collects the number of forward passes per
        pipeline inference, self.counter returns a sum of all the calls

        :return: The total amount of times the logger has been called by
            any pipeline
        """
        return sum(self._counter.values())

    def log_latency(
        self,
        pipeline_name: str,
        inference_timing: "InferenceTimingSchema",  # noqa F821
    ):
        """
        Continuously logs the inference pipeline latencies

        :param pipeline_name: The name of the inference pipeline from which the
            logger consumes the inference information to be monitored
        :param inference_timing: Pydantic model that contains information
            about time deltas of various processes within the inference pipeline
        """
        if pipeline_name not in self.metrics:
            # will be run on the first digestion of
            # inference_timing data for the given pipeline
            self._setup_metrics(pipeline_name, inference_timing)

        self._assert_incoming_data_consistency(pipeline_name, inference_timing)

        for metric_name, value_to_log in dict(inference_timing).items():
            self._log_latency(
                metric_name=metric_name,
                value_to_log=value_to_log,
                pipeline_name=pipeline_name,
            )

        self._update_call_count(pipeline_name)
        self._export_metrics_to_textfile()

    def log_data(self, pipeline_name: str, inputs: Any, outputs: Any):
        """

        :param pipeline_name: The name of the inference pipeline from which the
            logger consumes the inference information to be monitored
        :param inputs: the data received and consumed by the inference
            pipeline
        :param outputs: the data returned by the inference pipeline
        """
        pass

    def _log_latency(self, metric_name: str, value_to_log: float, pipeline_name: str):
        """
        Logs the latency value of a given metric

        :param metric_name: Name of the metric
        :param value_to_log: Time delta to be logged [in seconds]
        """
        histogram = self.metrics[pipeline_name][metric_name]
        histogram.observe(value_to_log)

    def _setup_client(self):
        """
        Starts the Prometheus client
        """
        start_http_server(port=self.port)
        _LOGGER.info(f"Prometheus client: started. Using port: {self.port}.")

    def _setup_metrics(
        self, pipeline_name: str, inference_timing: "InferenceTimingSchema"  # noqa F821
    ):
        """
        Sets up the set of metrics that the logger will expect
        to receive continuously

        :param inference_timing: Pydantic model that contains information
            about time deltas processes within the inference pipeline
        """
        # set multiple Histograms to track latencies
        # per Prometheus docs:
        # Histograms track the size and number of events in buckets
        for field_name, field_data in inference_timing.__fields__.items():
            field_description = field_data.field_info.description
            metric_name = (
                f"{pipeline_name}:{field_name}".strip()
                .replace(" ", "-")
                .replace("-", "_")
            )
            self.metrics[pipeline_name][field_name] = Histogram(
                metric_name, field_description, registry=REGISTRY
            )
        _LOGGER.info(
            f"Prometheus client: set the metrics to track pipeline: {pipeline_name}. "
            f"Added metrics: {[metric for metric in self.metrics[pipeline_name]]}"
        )

    def _export_metrics_to_textfile(self):
        if self.counter % self.text_log_save_freq == 0:
            write_to_textfile(self.text_logs_path, REGISTRY)

    def _update_call_count(self, pipeline_name: str):
        if pipeline_name in self._counter:
            self._counter[pipeline_name] += 1
        else:
            self._counter[pipeline_name] = 1

    def _assert_incoming_data_consistency(
        self, pipeline_name: str, inference_timing: "InferenceTimingSchema"  # noqa F821
    ):
        # once the metrics for the given pipepline have been setup,
        # make sure that the incoming `inference_timing` data
        # is consistent with the expected tracked metrics
        pipeline_metrics = self.metrics[pipeline_name]
        expected_metrics_names = {metric_name for metric_name in pipeline_metrics}
        received_metric_names = set(list(dict(inference_timing).keys()))
        if expected_metrics_names != received_metric_names:
            raise ValueError(
                f"Prometheus client: the received metrics "
                f"for the pipeline: {pipeline_name} "
                "are different from the expected. "
                f"Expected to receive metrics: {expected_metrics_names}, "
                f"but received {received_metric_names}"
            )
