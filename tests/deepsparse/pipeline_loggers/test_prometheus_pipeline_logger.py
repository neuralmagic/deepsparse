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

import requests

from deepsparse.pipeline_loggers import PrometheusLogger
from deepsparse.timing.timing_schema import InferenceTimingSchema


def test_prometheus_pipeline_logger(no_iterations=5.0, port=8000):
    args = {
        "pre_process_delta": 0.1,
        "engine_forward_delta": 0.2,
        "post_process_delta": 0.3,
        "total_inference_delta": 0.6,
    }

    logger = PrometheusLogger(port=port)
    for iteration in range(int(no_iterations)):
        args = {name: value + iteration for (name, value) in args.items()}
        inference_timing = InferenceTimingSchema(**args)
        logger.log_latency(inference_timing)

    # fetch the metrics from the server and validate their content
    response = requests.get(f"http://0.0.0.0:{port}/metrics").text
    response_lines = [x for x in response.split("\n")]
    for name, value in args.items():
        assert any([f"{name}_count {no_iterations}" in line for line in response_lines])
