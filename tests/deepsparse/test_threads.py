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

import concurrent.futures
import time

import pytest
from deepsparse import ThreadPool


def _heavy_compute(compute_time=0.25):
    time.sleep(compute_time)


@pytest.fixture()
def threadpool():
    """ "
    An auto-delete fixture that yields a ThreadPool object to test
    """
    yield ThreadPool()


@pytest.mark.parametrize(
    "compute_times",
    [
        [0.25, 0.5, 0.75, 1],
        [0.02, 0.05, 0.075, 0.1],
        [0.1, 0.05, 0.3],
    ],
)
def test_submit_is_asynchronous(threadpool, compute_times):
    threadpool = ThreadPool()

    start = time.time()
    futures = [
        threadpool.submit(lambda: _heavy_compute(compute_time=compute_time))
        for compute_time in compute_times
    ]
    concurrent.futures.wait(futures)
    elapsed_async = time.time() - start

    start = time.time()
    for compute_time in compute_times:
        _heavy_compute(compute_time=compute_time)
    elapsed_sync = time.time() - start

    assert elapsed_async > max(compute_times), (
        "Async execution time can-not be " "less than most expensive task"
    )
    assert elapsed_sync > sum(compute_times), (
        "Sequential execution time must be " "more than sum(all individual tasks)"
    )
    assert elapsed_async < elapsed_sync, "Async execution slower than sequential"
