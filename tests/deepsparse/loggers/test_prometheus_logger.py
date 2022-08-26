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

import os
import socket

import requests

import pytest
from deepsparse.loggers.prometheus_logger import PrometheusLogger
from deepsparse.timing.timing_schema import InferenceTimingSchema


class Pipeline:
    def __init__(self, name):
        self.name = name
        self.schema_args = {
            "pre_process": 0.1,
            "engine_forward": 0.2,
            "post_process": 0.3,
            "total_inference": 0.6,
        }

    def run_with_monitoring(self):
        results, timings, data = None, InferenceTimingSchema(**self.schema_args), None
        return results, timings, data


def _find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 0))
    portnum = s.getsockname()[1]
    s.close()

    return portnum


@pytest.mark.parametrize(
    "no_iterations, pipeline_names",
    [(6, ["earth", "namek"]), (10, ["moon"])],
    scope="class",
)
class TestPrometheusPipelineLogger:
    @pytest.fixture()
    def setup(self, tmp_path_factory, no_iterations, pipeline_names):
        port = _find_free_port()
        logger = PrometheusLogger(
            text_log_save_dir=tmp_path_factory.mktemp("logs"),
            # logs for each pipeline will be dumped after
            # all the iterations are finished
            text_log_save_freq=no_iterations,
            port=port,
        )
        pipelines = [Pipeline(name) for name in pipeline_names]
        yield logger, pipelines, no_iterations, port

    def test_log_latency(self, setup):
        logger, pipelines, no_iterations, port = setup

        for pipeline in pipelines:
            for _ in range(no_iterations):
                results, timings, data = pipeline.run_with_monitoring()
                logger.log_latency(pipeline.name, timings)

        self._check_logs(
            logger=logger,
            pipelines=pipelines,
            no_iterations=no_iterations,
            timings=timings,
            port=port,
        )

    @staticmethod
    def _check_logs(logger, pipelines, no_iterations, timings, port):

        # make sure that the text logs are created
        text_logs_path = logger.text_logs_path
        assert os.path.exists(text_logs_path)
        with open(text_logs_path) as f:
            text_logs_lines = f.readlines()
        assert text_logs_lines
        for pipeline in pipelines:
            TestPrometheusPipelineLogger._check_correct_count(
                text_logs_lines, timings, pipeline, no_iterations
            )

        # make sure that the metrics from the server are properly created
        response = requests.get(f"http://0.0.0.0:{port}").text
        request_log_lines = [x for x in response.split("\n")]
        assert request_log_lines
        for pipeline in pipelines:
            TestPrometheusPipelineLogger._check_correct_count(
                request_log_lines, timings, pipeline, no_iterations
            )

    @staticmethod
    def _check_correct_count(lines, timings, pipeline, no_iterations):
        for name, value in dict(timings).items():
            searched_line = f"{name}_{pipeline.name}_count {float(no_iterations)}"
            assert any([searched_line in line for line in lines])
