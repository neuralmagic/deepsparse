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

from deepsparse import FunctionLogger, Pipeline, PrometheusLogger
from tests.helpers import find_free_port
from tests.utils import mock_engine


@mock_engine(rng_seed=0)
def test_python_logger(engine, capsys, tmp_path):
    config = {
        "target": "pipeline_inputs",
        "mappings": [
            {
                "func": "tests/deepsparse/loggers/test_data/metric_functions.py:return_one",
                "frequency": 2,
            }
        ],
    }

    port = find_free_port()
    pipeline = Pipeline.create(
        "token_classification",
        batch_size=1,
        logger=FunctionLogger(
            PrometheusLogger(
                port=port, text_log_save_frequency=1, text_log_save_dir=tmp_path
            ),
            config=config,
        ),
    )

    for _ in range(20):
        pipeline("all_your_base_are_belong_to_us")
    response = requests.get(f"http://0.0.0.0:{port}").text
    request_log_lines = [x for x in response.split("\n") if "pipeline_inputs" in x]
    count_request = float(request_log_lines[2].split(" ")[1])

    text = None
    request_text_lines = None
    pass
