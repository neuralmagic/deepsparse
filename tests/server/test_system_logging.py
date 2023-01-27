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

from unittest import mock

import pytest
from deepsparse.loggers.config import SystemLoggingGroup
from deepsparse.server.config import (
    EndpointConfig,
    ServerConfig,
    ServerSystemLoggingConfig,
)
from deepsparse.server.helpers import server_logger_from_config
from deepsparse.server.server import _build_app
from deepsparse.server.system_logging import log_resource_utilization
from fastapi.testclient import TestClient
from tests.deepsparse.loggers.helpers import ListLogger
from tests.utils import mock_engine


logger_identifier = "tests/deepsparse/loggers/helpers.py:ListLogger"
stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni"  # noqa E501
task = "text-classification"
name = "endpoint_name"


def _test_successful_requests(calls, successful_request):
    relevant_call = [
        call
        for call in calls
        if call.startswith("identifier:request_details/successful_request_count")
    ]
    assert len(relevant_call) == 1
    relevant_call = relevant_call[0]
    value = bool(int(relevant_call.split("value:")[1].split(",")[0]))
    assert value == successful_request


def _test_input_batch_size(calls, input_batch_size):
    relevant_call = [
        call
        for call in calls
        if call.startswith(
            "identifier:endpoint_name/request_details/input_batch_size_total"
        )
    ]
    assert len(relevant_call) == 1
    relevant_call = relevant_call[0]
    value = int(relevant_call.split("value:")[1].split(",")[0])
    assert value == input_batch_size


def _test_response_msg(calls, response_msg):
    relevant_call = [
        call
        for call in calls
        if call.startswith("identifier:request_details/response_message")
    ]
    assert len(relevant_call) == 1
    relevant_call = relevant_call[0]
    value = relevant_call.split("value:")[1].split(",")[0]
    assert value == response_msg


@pytest.mark.parametrize(
    "json_payload, input_batch_size, successful_request, response_msg",
    [
        ({"sequences": "today is great"}, 1, True, "Response status code: 200"),
        (
            {"sequences": ["today is great", "today is great"]},
            2,
            True,
            "Response status code: 200",
        ),
        ({"this": "is supposed to fail"}, 1, False, "Response status code: 422"),
    ],
)
def test_log_request_details(
    json_payload, input_batch_size, successful_request, response_msg
):
    server_config = ServerConfig(
        endpoints=[
            EndpointConfig(
                task=task, name=name, model=stub, batch_size=input_batch_size
            )
        ],
        loggers={"logger_1": {"path": logger_identifier}},
        system_logging=ServerSystemLoggingConfig(
            request_details=SystemLoggingGroup(enable=True)
        ),
    )
    server_logger = server_logger_from_config(server_config)
    with mock.patch(
        "deepsparse.server.server.server_logger_from_config", return_value=server_logger
    ), mock_engine(rng_seed=0):
        app = _build_app(server_config)
    client = TestClient(app)
    client.post("/predict", json=json_payload)

    calls = server_logger.logger.loggers[0].logger.loggers[0].calls

    _test_successful_requests(calls, successful_request)
    _test_response_msg(calls, response_msg)
    if successful_request:
        _test_input_batch_size(calls, input_batch_size)


def _test_cpu_utilization(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith("identifier:resource_utilization/cpu_utilization_percent")
    ]
    assert len(relevant_calls) == num_iterations


def _test_memory_utilization(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith("identifier:resource_utilization/memory_utilization_percent")
    ]
    values = [float(call.split("value:")[1].split(",")[0]) for call in relevant_calls]
    assert len(relevant_calls) == num_iterations
    # memory utilization is a percentage, so it should be between 0 and 100
    assert all(0.0 < value < 100.0 for value in values)


def _test_total_memory_available(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith(
            "identifier:resource_utilization/total_memory_available_bytes"
        )
    ]
    values = [float(call.split("value:")[1].split(",")[0]) for call in relevant_calls]
    assert len(relevant_calls) == num_iterations
    # assert all values are the same (total memory available is constant)
    assert all(value == values[0] for value in values)


def _test_additional_items_to_log(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith("identifier:resource_utilization/test")
    ]
    values = [float(call.split("value:")[1].split(",")[0]) for call in relevant_calls]
    assert len(relevant_calls) == num_iterations
    # assert all values are the same ({"test" : 1} is constant)
    assert all(value == 1 for value in values)


@pytest.mark.parametrize(
    "num_iterations, additional_items_to_log",
    [
        (5, {}),
        (3, {"test": 1}),
    ],
)
def test_log_resource_utilization(num_iterations, additional_items_to_log):
    server_logger = ListLogger()

    for iter in range(num_iterations):
        log_resource_utilization(
            server_logger, prefix="resource_utilization", **additional_items_to_log
        )

    calls = server_logger.calls

    _test_cpu_utilization(calls, num_iterations)
    _test_memory_utilization(calls, num_iterations)
    _test_total_memory_available(calls, num_iterations)
