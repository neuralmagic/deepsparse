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

from deepsparse.utils import InferenceStages, StagedTimer, TimerManager


def test_staged_timer():
    timer = StagedTimer(enabled=True)

    timer.start(InferenceStages.ENGINE_FORWARD)
    time.sleep(1)  # sleep for 1 second to measure time
    timer.stop(InferenceStages.ENGINE_FORWARD)

    times = timer.times
    all_times = timer.all_times

    assert InferenceStages.ENGINE_FORWARD in times
    assert (
        0.9 <= times[InferenceStages.ENGINE_FORWARD] <= 1.1
    )  # account for minor time differences
    assert InferenceStages.ENGINE_FORWARD in all_times
    assert len(all_times[InferenceStages.ENGINE_FORWARD]) == 1
    assert (
        0.9 <= all_times[InferenceStages.ENGINE_FORWARD][0] <= 1.1
    )  # account for minor time differences


def test_timer_manager():
    timer_manager = TimerManager(enabled=True, multi=True)

    with timer_manager.new_timer_context() as timer:
        timer.start(InferenceStages.ENGINE_FORWARD)
        time.sleep(1)  # sleep for 1 second to measure time
        timer.stop(InferenceStages.ENGINE_FORWARD)

    times = timer_manager.times
    all_times = timer_manager.all_times

    assert InferenceStages.ENGINE_FORWARD in times
    assert (
        0.9 <= times[InferenceStages.ENGINE_FORWARD] <= 1.1
    )  # account for minor time differences
    assert InferenceStages.ENGINE_FORWARD in all_times
    assert len(all_times[InferenceStages.ENGINE_FORWARD]) == 1
    assert (
        0.9 <= all_times[InferenceStages.ENGINE_FORWARD][0] <= 1.1
    )  # account for minor time differences


def test_timer_manager_multithreaded():
    timer_manager = TimerManager(enabled=True, multi=True)

    def nested_func():
        timer = timer_manager.current
        assert timer is not None
        timer.start(InferenceStages.POST_PROCESS)
        time.sleep(1)  # sleep for 1 second to measure time
        timer.stop(InferenceStages.POST_PROCESS)

    def worker():
        with timer_manager.new_timer_context() as timer:
            timer.start(InferenceStages.ENGINE_FORWARD)
            time.sleep(1)  # sleep for 1 second to measure time
            timer.stop(InferenceStages.ENGINE_FORWARD)
            nested_func()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(worker)
        executor.submit(worker)

    times = timer_manager.times
    all_times = timer_manager.all_times

    # Checks for ENGINE_FORWARD stage
    assert InferenceStages.ENGINE_FORWARD in times
    assert (
        0.9 <= times[InferenceStages.ENGINE_FORWARD] <= 1.1
    )  # account for minor time differences
    assert InferenceStages.ENGINE_FORWARD in all_times
    assert len(all_times[InferenceStages.ENGINE_FORWARD]) == 2
    for t in all_times[InferenceStages.ENGINE_FORWARD]:
        assert 0.9 <= t <= 1.1  # account for minor time differences

    # Checks for POST_PROCESS stage
    assert InferenceStages.POST_PROCESS in times
    assert (
        0.9 <= times[InferenceStages.POST_PROCESS] <= 1.1
    )  # account for minor time differences
    assert InferenceStages.POST_PROCESS in all_times
    assert len(all_times[InferenceStages.POST_PROCESS]) == 2
    for t in all_times[InferenceStages.POST_PROCESS]:
        assert 0.9 <= t <= 1.1  # account for minor time differences
