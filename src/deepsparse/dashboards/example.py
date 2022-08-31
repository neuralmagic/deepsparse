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

import random

from deepsparse.loggers import PrometheusLogger
from deepsparse.timing import InferenceTimingSchema


PIPELINE_NAME = "sample_pipeline_name"
PYTHON_CLIENT_PORT = 8000  # port that exposes internal metrics via an HTTP endpoint


def get_inference_timing():
    timing_args = {
        "pre_process": random.random(),
        "engine_forward": random.random(),
        "post_process": random.random(),
    }
    timing_args["total_inference"] = sum(timing_args.values())
    return InferenceTimingSchema(**timing_args)


logger = PrometheusLogger(port=PYTHON_CLIENT_PORT, grafana_monitoring=True)

while True:
    inference_timing = get_inference_timing()
    logger.log_latency(PIPELINE_NAME, inference_timing)
