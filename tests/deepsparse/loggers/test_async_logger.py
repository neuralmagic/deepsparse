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
import time
from pathlib import Path

import numpy

import pytest
from deepsparse.loggers import AsyncLogger, FunctionLogger, MetricCategories
from tests.deepsparse.loggers.helpers import (
    ErrorLogger,
    FileLogger,
    NullLogger,
    SleepLogger,
)


def _build_sleep_logger():
    return SleepLogger(logger=NullLogger(), sleep_time=0.3)


def _build_numpy_function_logger():
    return FunctionLogger(
        logger=NullLogger(),
        target_identifier="test_log",
        function_name="mean",
        function=numpy.mean,
        frequency=1,
    )


@pytest.mark.parametrize(
    "base_logger_lambda,logged_value_lambda",
    [
        # tests that a logger that sleeps for 1s does not block
        (_build_sleep_logger, lambda: None),
        # tests 1) numpy functions and arrays may be submitted to async backend
        # 2) even if logged value is large, submission does not significantly
        # block (relevant if executor changes and requires serialization/IPC)
        (_build_numpy_function_logger, lambda: numpy.random.randn(int(3e7))),
    ],
)
def test_async_log_is_non_blocking(base_logger_lambda, logged_value_lambda):
    logger = AsyncLogger(logger=base_logger_lambda())
    logged_value = logged_value_lambda()

    # allow for max elapsed time of 0.1 ms
    _log_and_test_time_elasped(logger, logged_value, 0.1)


def test_async_log_executes(tmp_path: Path):
    log_file_path = tmp_path / "test_logs.txt"
    logger = AsyncLogger(logger=FileLogger(file_path=log_file_path))

    with open(log_file_path, "r") as file:
        # base check that no logs have been written
        assert len(file.readlines()) == 0

    # log sample value to file
    logged_value = "test_log_value"
    _log_and_test_time_elasped(logger, logged_value, 0.1)

    # sleep to allow log job to complete then check file has been updated
    time.sleep(0.5)
    with open(log_file_path, "r") as file:
        # base check that no logs have been written
        logged_items = file.readlines()

    assert len(logged_items) == 1
    assert logged_value in logged_items[0]


def test_async_logger_error_propagation(caplog):
    logger = AsyncLogger(logger=ErrorLogger())

    # listen for expected ERROR log
    caplog.set_level(logging.ERROR)
    assert len(caplog.messages) == 0  # check no logs have been issued yet

    logger.log("test_log", 1.0, MetricCategories.SYSTEM)
    time.sleep(0.5)

    assert len(caplog.messages) == 1
    assert "RuntimeError('Raising for testing purposes')" in caplog.messages[0]


def _log_and_test_time_elasped(logger, logged_value, max_time_elapsed_ms):
    log_call_start = time.time()
    logger.log("test_log", logged_value, MetricCategories.SYSTEM)
    log_call_time_ms = time.time() - log_call_start

    assert log_call_time_ms <= 0.1
