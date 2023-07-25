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
import string
import logging
import random
from typing import Dict, List, Tuple
import time
import numpy
import threading
import queue

from deepsparse import __version__
from deepsparse import Pipeline
from deepsparse.cpu import cpu_architecture
from deepsparse.log import set_logging_level
from deepsparse.utils.timer import StagedTimer
from deepsparse.benchmark.helpers import (
    decide_thread_pinning,
    parse_scheduler,
    parse_scenario,
    parse_num_streams
)


__all__ = ["benchmark_pipeline"]


_LOGGER = logging.getLogger(__name__)

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSparse Pipelines"
    )
    parser.add_argument(
        "task_name",
        type=str,
        help="Type of pipeline to run"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to an ONNX model file or SparseZoo model stub",
    )
    parser.add_argument(
        "-c",
        "--input_config",
        type=str,
        default="config.json",
        help="JSON file containing schema for input data"
    )
    parser.add_argument(
        "-i",
        "--input_type",
        type=str,
        default="dummy",
        choices=["dummy", "real"],
        help="Type of input data to use, real or randomly generated"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to run the analysis for. Must be greater than 0",
    )
    parser.add_argument(
        "-ncores",
        "--num_cores",
        type=int,
        default=cpu_architecture().num_available_physical_cores,
        help=(
            "The number of physical cores to run the analysis on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        default="sync",
        choices=["async", "sync", "elastic"],
        help=(
            "Choose between using the async, sync and elastic scenarios. Sync and "
            "async are similar to the single-stream/multi-stream scenarios. Elastic "
            "is a newer scenario that behaves similarly to the async scenario "
            "but uses a different scheduling backend. Default value is sync."
        ),
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=10,
        help="The number of seconds the benchmark will run. Default is 10 seconds.",
    )
    parser.add_argument(
        "-nstreams",
        "--num_streams",
        type=int,
        default=None,
        help=(
            "The number of streams that will submit inferences in parallel using "
            "async scenario. Default is automatically determined for given hardware "
            "and may be sub-optimal."
        ),
    )
    parser.add_argument(
        "-pin",
        "--thread_pinning",
        type=str,
        default="core",
        choices=["none", "core", "numa"],
        help=(
            "Enable binding threads to cores ('core' the default), "
            "threads to cores on sockets ('numa'), or disable ('none')"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Lower logging verbosity",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-x",
        "--export_path",
        help="Store results into a JSON file",
        type=str,
        default=None,
    )

    return parser.parse_args()

class PipelineExecutorThread(threading.Thread):
    def __init__(
        self,
        pipeline: Pipeline,
        inputs: List[any],
        time_queue: queue.Queue,
        max_time: float
    ):
        super(PipelineExecutorThread, self).__init__()
        self._pipeline = pipeline
        self._inputs = inputs
        self._time_queue = time_queue
        self._max_time = max_time

    def run(self):
        while time.perf_counter() < self._max_time:
            output = self._pipeline(self._inputs)
            self._time_queue.put(self._pipeline.timer_manager.latest)


def singlestream_benchmark(
    pipeline: Pipeline,
    inputs: List[any],
    seconds_to_run: float
) -> List[StagedTimer]:
    benchmark_end_time = time.perf_counter() + seconds_to_run
    batch_timings = []
    while time.perf_counter() < benchmark_end_time:
        output = pipeline(inputs)
        batch_timings.append(pipeline.timer_manager.latest)

    return batch_timings

def multistream_benchmark(
    pipeline: Pipeline,
    inputs: List[any],
    seconds_to_run: float,
    num_streams: int,
) -> List[StagedTimer]:
    time_queue = queue.Queue()
    max_time = time.perf_counter() + seconds_to_run
    threads = []

    # Sara TODO: should these all be sharing the same pipeline?
    for thread in range(num_streams):
        threads.append(PipelineExecutorThread(pipeline, inputs, time_queue, max_time))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return list(time_queue.queue)


def parse_input_config(input_config_file: str) -> Dict[str, any]:
    config_file = open(input_config_file)
    config = json.load(config_file)
    config_file.close()
    return config

def benchmark_pipeline(
    model_path: str,
    task: str,
    input_config: str,
    batch_size: int = 1,
    num_cores: int = None,
    scenario: str = "sync",
    seconds_to_run: int = 10,
    num_streams: int = None,
    thread_pinning: str = "core",
    quiet: bool = False,
) -> Tuple[List[StagedTimer],float] :
    
    if quiet:
        set_logging_level(logging.WARN)

    if num_cores is None:
        num_cores = cpu_architecture().num_available_physical_cores

    decide_thread_pinning(thread_pinning, _LOGGER)
    scenario = parse_scenario(scenario.lower(), _LOGGER)
    num_streams = parse_num_streams(num_streams, num_cores, scenario, _LOGGER)
    
    config = parse_input_config(input_config)
    input_type = config["data_type"]
    pipeline = Pipeline.create(task=task, model_path=model_path)

    input_data = []
    if input_type == "dummy":
        if config['input_data_type'] == "string":
            data_length = config['sequence_length']
            for _ in range(batch_size):
                rand_string = ''.join(random.choices(string.printable, k=data_length))
                input_data.append(rand_string)
            inputs = pipeline.input_schema(sequences=input_data)
        elif config['input_data_type'] == "array":
            image_shape = config["input_array_shape"]
            dtype = config["input_array_dtype"]
            for _ in range(batch_size):
                if dtype == "uint8":
                    rand_array = numpy.random.randint(0,high=255, size=image_shape).astype(dtype)
                rand_array = numpy.random.rand(*image_shape).astype(dtype)
                input_data.append(rand_array)
            inputs = pipeline.input_schema(images=input_data)
    elif input_type == "real":
        raise Exception("Real input type not yet implemented")
    else:
        raise Exception(f"Unknown input type '{input_type}'")


    start_time = time.perf_counter()
    if scenario == "singlestream":
        batch_times = singlestream_benchmark(pipeline, inputs, seconds_to_run)
    elif scenario == "multistream":
        batch_times = multistream_benchmark(pipeline, inputs, seconds_to_run, num_streams)
    elif scenario == "elastic":
        batch_times = multistream_benchmark(pipeline, inputs, seconds_to_run, num_streams)
    else:
        raise Exception(f"Unknown scenario '{scenario}'")

    if len(batch_times) == 0:
        raise Exception(
            "Generated no batch timings, try extending benchmark time with '--time'"
        )
    end_time = time.perf_counter()
    total_run_time = end_time - start_time

    return batch_times, total_run_time

def calculate_statistics(batch_times_ms: List[float]) -> Dict:
    percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    buckets = numpy.percentile(batch_times_ms, percentiles).tolist()
    percentiles_dict = {
        "{:2.1f}%".format(key): value for key, value in zip(percentiles, buckets)
    }

    benchmark_dict = {
        "median": numpy.median(batch_times_ms),
        "mean": numpy.mean(batch_times_ms),
        "std": numpy.std(batch_times_ms),
        **percentiles_dict,
    }
    return benchmark_dict

def main():
    args = parse_args()

    print("Original Model Path: {}".format(args.model_path))
    print("Task: {}".format(args.task_name))
    print("Input Type: {}".format(args.input_type))
    print("Batch Size: {}".format(args.batch_size))
    print("Scenario: {}".format(args.scenario))

    batch_times, total_run_time = benchmark_pipeline(
        model_path=args.model_path,
        task=args.task_name,
        input_config = args.input_config,
        batch_size=args.batch_size,
        num_cores=args.num_cores,
        scenario=args.scenario,
        seconds_to_run=args.time,
        num_streams=args.num_streams,
        thread_pinning=args.thread_pinning,
        quiet=args.quiet,
    )

    pre_process_times = [st.times['pre_process'] * 1000 for st in batch_times]
    pre_stats = calculate_statistics(pre_process_times)
    post_process_times = [st.times['post_process'] * 1000 for st in batch_times]
    post_stats = calculate_statistics(post_process_times)
    engine_forward_times = [st.times['engine_forward'] * 1000 for st in batch_times]
    forward_stats = calculate_statistics(engine_forward_times)

    items_per_sec = (len(batch_times) * args.batch_size) / total_run_time

    total_pre_process = sum(pre_process_times)
    total_post_process = sum(post_process_times)
    total_engine_forward = sum(engine_forward_times)
    total_time = total_pre_process + total_post_process + total_engine_forward
    percent_pre = total_pre_process / total_time * 100
    percent_post = total_post_process / total_time * 100
    percent_forward = total_engine_forward / total_time * 100

    export_dict = {
        "scenario": args.scenario,
        "items_per_sec": items_per_sec,
        "seconds_ran": total_run_time,
        "iterations": len(batch_times),
        "percent_pre": percent_pre,
        "percent_post": percent_post,
        "percent_forward": percent_forward,
        "pre_stats": pre_stats,
        "post_stats": post_stats,
        "forward_stats": forward_stats
    }

    # Export results
    export_path = args.export_path
    if export_path:
        _LOGGER.info("Saving benchmark results to JSON file at {}".format(export_path))
        with open(export_path, "w") as out:
            json.dump(export_dict, out, indent=2)

    # Results summary
    print("Original Model Path: {}".format(args.model_path))
    print("Batch Size: {}".format(args.batch_size))
    print("Scenario: {}".format(args.scenario))
    print("Iterations: {}".format(int(export_dict["iterations"])))
    print(
        "Throughput (items/sec): {:.4f}".format(
            export_dict["items_per_sec"]
        )
    )
    print("Processing Time Breakdown: ")
    print("     Pre-Processing: {:.2f}%".format(export_dict["percent_pre"]))
    print("     Post-Processing: {:.2f}%".format(export_dict["percent_post"]))
    print("     Forward Pass: {:.2f}%".format(export_dict["percent_forward"]))
    print("Pre-Processing Latency Mean (ms/batch): {:.4f}".format(export_dict["pre_stats"]["mean"]))
    print("Post-Processing Latency Mean (ms/batch): {:.4f}".format(export_dict["post_stats"]["mean"]))
    print("Forward Pass Latency Mean (ms/batch): {:.4f}".format(export_dict["forward_stats"]["mean"]))

if __name__ == "__main__":
    main()
