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
from typing import Any

from deepsparse import PipelineLogger
from prometheus_client import Histogram, start_http_server


__all__ = ["PrometheusLogger"]

_LOGGER = logging.getLogger(__name__)


class PrometheusLogger(PipelineLogger):
    """
    PipelineLogger that uses the official Prometheus python client
    (https://github.com/prometheus/client_python) to monitor the
    inference pipeline

    :param port: the port used by the client. Default is 8000
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self.setup_client()
        # the data structure responsible for the instrumentation
        # of the metrics
        self.metrics = []

        super().__init__()

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

        _LOGGER.info("Prometheus client: logged latency metrics")

    def log_data(self, inputs: Any, outputs: Any):
        raise NotImplementedError()

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
            self.metrics.append(Histogram(field_name, field_description))
        _LOGGER.info(
            "Prometheus client: set the metrics to track. "
            f"Tracked metrics: {[metric for metric in self.metrics]}"
        )

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
