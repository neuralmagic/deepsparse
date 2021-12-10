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

import argparse
import json
import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from deepsparse import Engine, Scheduler, compile_model
from deepsparse.utils import generate_random_inputs


__all__ = ["model_stream_benchmark"]


def verify_output(outputs: List[np.ndarray], correct_outputs: List[np.ndarray]):
    assert len(correct_outputs) == len(outputs)
    for i in range(len(correct_outputs)):
        gt_output = correct_outputs[i]
        nm_output = outputs[i]

        assert gt_output.shape == nm_output.shape
        assert type(gt_output) == type(nm_output)

        max_diff = np.max(np.abs(nm_output - gt_output))
        if not np.allclose(nm_output, gt_output, 0.0, 8.0e-4):
            raise Exception(
                "ERROR: output {}: {} {} MAX DIFF: {}".format(
                    i, gt_output.shape, nm_output.shape, max_diff
                )
            )


def iteration(model: Engine, input: List[np.ndarray]):
    start = time.time()
    output = model(input)
    end = time.time()
    return output, start, end


def singlestream_benchmark(
    model: Engine,
    input_list: List[np.ndarray],
    output_list: Optional[List[np.ndarray]],
    seconds_to_run: float,
) -> List[float]:
    batch_times = []

    stream_end_time = time.time() + seconds_to_run
    while time.time() < stream_end_time:
        output, start, end = iteration(model, input_list)
        batch_times.append([start, end])
        if output_list:
            verify_output(output, output_list)

    return batch_times


class EngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model: Engine,
        input_list: List[np.ndarray],
        output_list: Optional[List[np.ndarray]],
        time_queue: queue.Queue,
        max_time: float,
    ):
        super(EngineExecutorThread, self).__init__()
        self._model = model
        self._input_list = input_list
        self._output_list = output_list
        self._time_queue = time_queue
        self._max_time = max_time

    def run(self):
        while time.time() < self._max_time:
            output, start, end = iteration(self._model, self._input_list)
            self._time_queue.put([start, end])
            if self._output_list:
                verify_output(output, self._output_list)


def multistream_benchmark(
    model: Engine,
    input_list: List[np.ndarray],
    output_list: Optional[List[np.ndarray]],
    seconds_to_run: float,
    num_streams: int,
) -> List[float]:
    time_queue = queue.Queue()
    max_time = time.time() + seconds_to_run
    threads = []

    for thread in range(num_streams):
        threads.append(
            EngineExecutorThread(model, input_list, output_list, time_queue, max_time)
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return list(time_queue.queue)


def model_stream_benchmark(
    model: Engine,
    input_list: List[np.ndarray],
    output_list: Optional[List[np.ndarray]],
    scenario: str,
    seconds_to_run: float,
    num_streams: int,
) -> Dict:
    batch_times = []

    # Run the benchmark scenario and collect batch times
    if scenario == "singlestream":
        batch_times = singlestream_benchmark(
            model, input_list, output_list, seconds_to_run
        )
    elif scenario == "multistream":
        batch_times = multistream_benchmark(
            model, input_list, output_list, seconds_to_run, num_streams
        )
    else:
        raise Exception(f"Unknown scenario '{scenario}'")

    # Convert times to milliseconds
    batch_times_ms = [
        (batch_time[1] - batch_time[0]) * 1000 for batch_time in batch_times
    ]

    # Calculate statistics
    first_start_time = min([b[0] for b in batch_times])
    last_end_time = max([b[1] for b in batch_times])
    total_time_executing = last_end_time - first_start_time

    items_per_sec = (model.batch_size * len(batch_times)) / total_time_executing

    percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    buckets = np.percentile(batch_times_ms, percentiles).tolist()
    percentiles_dict = {
        "{:2.1f}%".format(key): value for key, value in zip(percentiles, buckets)
    }
    benchmark_dict = {
        "scenario": scenario,
        "items_per_sec": items_per_sec,
        "seconds_ran": total_time_executing,
        "iterations": len(batch_times_ms),
        "median": np.median(batch_times_ms),
        "mean": np.mean(batch_times_ms),
        "std": np.std(batch_times_ms),
        **percentiles_dict,
    }
    return benchmark_dict
