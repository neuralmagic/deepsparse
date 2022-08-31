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

import queue
import threading
import time
from typing import Dict, List

import numpy

from deepsparse import Engine


__all__ = ["model_stream_benchmark"]


def iteration(model: Engine, input: List[numpy.ndarray]):
    start = time.perf_counter()
    output = model.run(input, val_inp=False)
    end = time.perf_counter()
    return output, start, end


def singlestream_benchmark(
    model: Engine,
    input_list: List[numpy.ndarray],
    seconds_to_run: float,
) -> List[float]:
    batch_times = []

    stream_end_time = time.perf_counter() + seconds_to_run
    while time.perf_counter() < stream_end_time:
        _, start, end = iteration(model, input_list)
        batch_times.append([start, end])

    return batch_times


class EngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model: Engine,
        input_list: List[numpy.ndarray],
        time_queue: queue.Queue,
        max_time: float,
    ):
        super(EngineExecutorThread, self).__init__()
        self._model = model
        self._input_list = input_list
        self._time_queue = time_queue
        self._max_time = max_time

    def run(self):
        while time.perf_counter() < self._max_time:
            _, start, end = iteration(self._model, self._input_list)
            self._time_queue.put([start, end])


def multistream_benchmark(
    model: Engine,
    input_list: List[numpy.ndarray],
    seconds_to_run: float,
    num_streams: int,
) -> List[float]:
    time_queue = queue.Queue()
    max_time = time.perf_counter() + seconds_to_run
    threads = []

    for thread in range(num_streams):
        threads.append(EngineExecutorThread(model, input_list, time_queue, max_time))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return list(time_queue.queue)


def model_stream_benchmark(
    model: Engine,
    input_list: List[numpy.ndarray],
    scenario: str,
    seconds_to_run: float,
    seconds_to_warmup: float,
    num_streams: int,
) -> Dict:

    # Run the benchmark scenario and collect batch times. The engine will be warmed up
    # for a few seconds first using "seconds_to_warmup"
    if scenario == "singlestream":
        singlestream_benchmark(model, input_list, seconds_to_warmup)
        batch_times = singlestream_benchmark(model, input_list, seconds_to_run)
    elif scenario == "multistream":
        multistream_benchmark(model, input_list, seconds_to_warmup, num_streams)
        batch_times = multistream_benchmark(
            model, input_list, seconds_to_run, num_streams
        )
    elif scenario == "elastic":
        multistream_benchmark(model, input_list, seconds_to_warmup, num_streams)
        batch_times = multistream_benchmark(
            model, input_list, seconds_to_run, num_streams
        )
    else:
        raise Exception(f"Unknown scenario '{scenario}'")

    if len(batch_times) == 0:
        raise Exception(
            "Generated no batch timings, try extending benchmark time with '--time'"
        )

    # Convert times to milliseconds
    batch_times_ms = [
        (batch_time[1] - batch_time[0]) * 1000 for batch_time in batch_times
    ]

    # Calculate statistics
    # Note: We want to know all of the executions that could be performed within a
    # given amount of wallclock time. This calculation as-is includes the test overhead
    # such as saving timing results for each iteration so it isn't a best-case but is a
    # realistic case.
    first_start_time = min([b[0] for b in batch_times])
    last_end_time = max([b[1] for b in batch_times])
    total_time_executing = last_end_time - first_start_time

    items_per_sec = (model.batch_size * len(batch_times)) / total_time_executing

    percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    buckets = numpy.percentile(batch_times_ms, percentiles).tolist()
    percentiles_dict = {
        "{:2.1f}%".format(key): value for key, value in zip(percentiles, buckets)
    }
    benchmark_dict = {
        "scenario": scenario,
        "items_per_sec": items_per_sec,
        "seconds_ran": total_time_executing,
        "iterations": len(batch_times_ms),
        "median": numpy.median(batch_times_ms),
        "mean": numpy.mean(batch_times_ms),
        "std": numpy.std(batch_times_ms),
        **percentiles_dict,
    }
    return benchmark_dict
