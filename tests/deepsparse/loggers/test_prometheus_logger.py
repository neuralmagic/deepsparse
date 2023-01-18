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
from deepsparse import PrometheusLogger
from deepsparse.loggers import MetricCategories
from tests.helpers import find_free_port
from tests.utils import mock_engine


@pytest.mark.parametrize(
    "identifier, no_iterations, value, text_log_save_frequency, should_fail",
    [
        ("dummy.identifier_1", 20, 1.0, 1, False),
        ("dummy.identifier.2", 20, 1, 5, False),
        ("dummy.identifier.3", 20, [1.0], 10, True),
    ],
)
@mock_engine(rng_seed=0)
def test_prometheus_logger(
    engine,
    tmp_path,
    identifier,
    no_iterations,
    value,
    text_log_save_frequency,
    should_fail,
):
    port = find_free_port()
    logger = PrometheusLogger(
        port=port,
        text_log_save_frequency=text_log_save_frequency,
        text_log_save_dir=tmp_path,
    )

    for idx in range(no_iterations):
        if should_fail:
            with pytest.raises(ValueError):
                logger.log(identifier, value, MetricCategories.SYSTEM)
                return
            return
        logger.log(identifier, value, MetricCategories.SYSTEM)

    response = requests.get(f"http://0.0.0.0:{port}").text
    request_log_lines = response.split("\n")
    # line 38 is where we get '{identifier}_count {no_iterations}'
    count_request_request = float(request_log_lines[38].split(" ")[1])

    with open(logger.text_log_file_path) as f:
        text_log_lines = f.readlines()
    count_request_text = float(text_log_lines[38].split(" ")[1])

    assert count_request_request == count_request_text == no_iterations


@pytest.mark.parametrize(
    "identifier, value, expected_logs",
    [
        (
            "dummy_identifier",
            {"foo": {"alice": 1, "bob": 2}, "bar": 5},
            {
                "dummy_identifier__foo__alice_count 1.0",
                "dummy_identifier__foo__bob_count 1.0",
                "dummy_identifier__bar_count 1.0",
            },
        ),
    ],
)
@mock_engine(rng_seed=0)
def test_nested_value_inputs(engine, identifier, value, expected_logs):
    port = find_free_port()
    logger = PrometheusLogger(port=port)
    logger.log(identifier, value, MetricCategories.SYSTEM)
    response = requests.get(f"http://0.0.0.0:{port}").text
    request_log_lines = response.split("\n")
    assert set(request_log_lines).issuperset(expected_logs)
