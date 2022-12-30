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
import time

import pytest
from deepsparse.loggers import AsyncLogger, FunctionLogger, MultiLogger
from deepsparse.server.system_logging import log_resource_utilization
from tests.deepsparse.loggers.helpers import ListLogger


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
    # memory utilization is a percentage, so it should be between 0 and 1
    assert all(0.0 < value < 1.0 for value in values)


def _test_total_memory_available(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith("identifier:resource_utilization/total_memory_available_MB")
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
    # assert all values are the same (total memory available is constant)
    assert all(value == 1 for value in values)


@pytest.mark.parametrize(
    "num_iterations, target_identifier, additional_items_to_log",
    [
        (5, "resource_utilization", {}),
        (5, "resource_utilization", {"test": 1}),
        (5, "invalid", {}),
    ],
)
def test_log_resource_utilization(
    num_iterations, target_identifier, additional_items_to_log
):
    server_logger = AsyncLogger(
        logger=MultiLogger(
            [
                FunctionLogger(
                    logger=ListLogger(),
                    target_identifier=target_identifier,
                    function=lambda x: x,
                    function_name="identity",
                )
            ]
        ),
        max_workers=1,
    )

    for iter in range(num_iterations):
        log_resource_utilization(server_logger, **additional_items_to_log)

    time.sleep(1)
    if "resource_utilization" == target_identifier:
        _test_cpu_utilization(
            server_logger.logger.loggers[0].logger.calls, num_iterations
        )
        _test_memory_utilization(
            server_logger.logger.loggers[0].logger.calls, num_iterations
        )
        _test_total_memory_available(
            server_logger.logger.loggers[0].logger.calls, num_iterations
        )
        if additional_items_to_log:
            _test_additional_items_to_log(
                server_logger.logger.loggers[0].logger.calls, num_iterations
            )
    else:
        assert server_logger.logger.loggers[0].logger.calls == []
