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
from typing import Any, Optional

from deepsparse.pipeline_logger import PipelineLogger
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Histogram,
    start_http_server,
    write_to_textfile,
)


__all__ = ["PrometheusLogger"]

_LOGGER = logging.getLogger(__name__)

IDENTIFIER = "prometheus"


class PrometheusLogger(PipelineLogger):
    """
    PipelineLogger that uses the official Prometheus python client
    (https://github.com/prometheus/client_python) to monitor the
    inference pipeline

    :param pipeline_name: The name of the pipeline the
        logger refers to
    :param port: the port used by the client. Default is 8000
    :param text_log_save_dir: the directory where the text log files
        are saved. By default, the python working directory
    :param text_log_save_freq: the frequency of saving the text log
        files. E.g. if `text_log_save_freq` = 10, text logs are dumped
        after every tenth forward pass
    :param text_log_file_name: the name of the text log file.
        Default: `prometheus_logs.prom`.
    """

    def __init__(
        self,
        pipeline_name: Optional[str] = None,
        port: int = 8000,
        text_log_save_dir: str = os.getcwd(),
        text_log_save_freq: int = 10,
        text_log_file_name: Optional[str] = None,
    ):
        self.port = port
        self.text_log_save_freq = text_log_save_freq
        self.text_log_file_path = os.path.join(
            text_log_save_dir, text_log_file_name or f"{IDENTIFIER}_logs.prom"
        )
        self.setup_client()
        self._registry = CollectorRegistry()
        REGISTRY.register(self._registry)

        # the data structure responsible for the instrumentation
        # of the metrics
        self.metrics = []
        # track how many times the logger has been called
        self.counter = 0

        super().__init__(pipeline_name=pipeline_name, identifier=IDENTIFIER)

    def setup_client(self):
        """
        Starts the Prometheus client
        """
        start_http_server(port=self.port)
        _LOGGER.info(f"Prometheus client: started. Using port: {self.port}.")

    def log_latency(
        self, inference_timing: "InferenceTimingSchema", **kwargs  # noqa F821
    ):
        """
        Continuously logs the inference pipeline latencies

        :param inference_timing: Pydantic model that contains information
            about time deltas processes within the inference pipeline
        """
        if not self.metrics:
            # will be run on the first digestion of
            # inference_timing data
            self._setup_metrics(inference_timing)

        self._assert_incoming_data_consistency(inference_timing)

        for metric_name, value_to_log in dict(inference_timing).items():
            self._log_latency(metric_name=metric_name, value_to_log=value_to_log)

        self.counter += 1
        if self.counter % self.text_log_save_freq:
            self._export_metrics_to_textfile()

    def log_data(self, inputs: Any, outputs: Any):
        raise NotImplementedError()

    def _log_latency(self, metric_name: str, value_to_log: float):
        """
        Logs the latency value of a given metric

        :param metric_name: Name of the metric
        :param value_to_log: Time delta to be logged [in seconds]
        """
        histogram = [
            histogram for histogram in self.metrics if histogram._name == metric_name
        ][0]
        histogram.observe(value_to_log)

    def _setup_metrics(self, inference_timing: "InferenceTimingSchema"):  # noqa F821
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
            self.metrics.append(
                Histogram(field_name, field_description, registry=REGISTRY)
            )
        _LOGGER.info(
            "Prometheus client: set the metrics to track. "
            f"Tracked metrics: {[metric for metric in self.metrics]}"
        )

    def _export_metrics_to_textfile(self):
        write_to_textfile(self.text_log_file_path, REGISTRY)

    def _update_call_count(self):
        call_counter = [
            metric for metric in self.metrics if metric._name == "num_forward_passes"
        ][0]
        call_counter.inc()

    def _assert_incoming_data_consistency(
        self, inference_timing: "InferenceTimingSchema"  # noqa F821
    ):
        # once the metrics have been setup,
        # make sure that the incoming `inference_timing` data
        # is consistent with the expected tracked metrics
        expected_metrics_names = {metric._name for metric in self.metrics}
        received_metric_names = set(list(dict(inference_timing).keys()))
        if expected_metrics_names != received_metric_names:
            raise ValueError(
                "Prometheus client: the received metrics "
                "are different from the expected. "
                f"Expected to receive metrics: {expected_metrics_names}, "
                f"but received {received_metric_names}"
            )
