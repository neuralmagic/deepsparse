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
import pytest
from deepsparse.server.system_logging import log_resource_utilization
from tests.deepsparse.loggers.helpers import ListLogger


def _test_cpu_utilization(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith("identifier:resource_utilization/cpu_utilization_[%]")
    ]
    assert len(relevant_calls) == num_iterations


def _test_memory_utilization(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith("identifier:resource_utilization/memory_utilization_[%]")
    ]
    values = [float(call.split("value:")[1].split(",")[0]) for call in relevant_calls]
    assert len(relevant_calls) == num_iterations
    # memory utilization is a percentage, so it should be between 0 and 1
    assert all(0.0 < value < 1.0 for value in values)


def _test_total_memory_available(calls, num_iterations):
    relevant_calls = [
        call
        for call in calls
        if call.startswith(
            "identifier:resource_utilization/total_memory_available_[MB]"
        )
    ]
    values = [float(call.split("value:")[1].split(",")[0]) for call in relevant_calls]
    assert len(relevant_calls) == num_iterations
    # assert all values are the same (total memory available is constant)
    assert all(value == values[0] for value in values)


def _test_kwargs(calls, num_iterations):
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
    "num_iterations, kwargs",
    [
        (5, {}),
        (5, {"test": 1}),
    ],
)
def test_log_resource_utilization(num_iterations, kwargs):
    logger = ListLogger()
    for iter in range(num_iterations):
        log_resource_utilization(logger, kwargs)

    _test_cpu_utilization(logger.calls, num_iterations)
    _test_memory_utilization(logger.calls, num_iterations)
    _test_total_memory_available(logger.calls, num_iterations)
    if kwargs:
        _test_kwargs(logger.calls, num_iterations)
