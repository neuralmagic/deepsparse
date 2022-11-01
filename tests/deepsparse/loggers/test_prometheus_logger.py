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

import pytest
from deepsparse import FunctionLogger, Pipeline, PrometheusLogger
from tests.helpers import find_free_port
from tests.utils import mock_engine


CONFIG = {
    "target": "pipeline_inputs",
    "mappings": [
        {
            "func": "tests/deepsparse/loggers/test_data/metric_functions.py:return_one",
            "frequency": 2,
        }
    ],
}

PORT = find_free_port()


@pytest.mark.parametrize(
    "config,port, no_iterations, expected_log_count, target",
    [(CONFIG, PORT, 20, 10, CONFIG["target"])],
)
@mock_engine(rng_seed=0)
def test_python_logger(
    engine, capsys, tmp_path, config, port, no_iterations, expected_log_count, target
):
    logger = FunctionLogger(
        PrometheusLogger(
            port=port, text_log_save_frequency=1, text_log_save_dir=tmp_path
        ),
        config=config,
    )
    pipeline = Pipeline.create("token_classification", batch_size=1, logger=logger)

    for _ in range(no_iterations):
        pipeline("all_your_base_are_belong_to_us")

    response = requests.get(f"http://0.0.0.0:{port}").text
    request_log_lines = [x for x in response.split("\n") if target in x]
    count_request_request = float(request_log_lines[2].split(" ")[1])

    with open(pipeline.logger.logger.text_log_file_path) as f:
        text_log_lines = f.readlines()
    text_log_lines = [x for x in text_log_lines if target in x]
    count_request_text = float(text_log_lines[2].split(" ")[1])

    assert count_request_request == count_request_text
