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
import logging
import multiprocessing as mp
import os

import numpy as np

import numa
from deepsparse import Scheduler, compile_model
from deepsparse.benchmark_model.stream_benchmark import singlestream_benchmark
from deepsparse.utils import (
    generate_random_inputs,
    model_to_path,
    override_onnx_input_shapes,
    parse_input_shapes,
)
from sparsezoo.models import Zoo


_LOGGER = logging.getLogger(__name__)

DEEPSPARSE_ENGINE = "deepsparse"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models in the DeepSparse Engine"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to an ONNX model file or SparseZoo model stub",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to run the analysis for. Must be greater than 0",
    )
    parser.add_argument(
        "-shapes",
        "--input_shapes",
        type=str,
        default="",
        help="Override the shapes of the inputs, "
        'i.e. -shapes "[1,2,3],[4,5,6],[7,8,9]" results in '
        "input0=[1,2,3] input1=[4,5,6] input2=[7,8,9]",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=10,
        help="The number of seconds the benchmark will run. Default is 10 seconds.",
    )
    parser.add_argument(
        "-w",
        "--warmup_time",
        type=int,
        default=2,
        help=(
            "The number of seconds the benchmark will warmup before running and cooldown after running."
            "Default is 2 seconds."
        ),
    )
    parser.add_argument(
        "-nstreams",
        "--num_streams",
        type=int,
        help=("The number of processes that will run inferences in parallel. "),
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

    return parser.parse_args()


def decide_thread_pinning(pinning_mode: str):
    pinning_mode = pinning_mode.lower()

    if pinning_mode in "core":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "1"
        _LOGGER.info("Thread pinning to cores enabled")
    elif pinning_mode in "numa":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "1"
        _LOGGER.info("Thread pinning to socket/numa nodes enabled")
    elif pinning_mode in "none":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "0"
        _LOGGER.info("Thread pinning disabled, performance may be sub-optimal")
    else:
        _LOGGER.info(
            "Recieved invalid option for thread_pinning '{}', skipping".format(
                pinning_mode
            )
        )


def cpus_to_numas(cpu_set):
    numas = set()
    for i in range(numa.info.get_max_node()):
        cpus = numa.info.node_to_cpus(i)
        for j in cpus:
            if j in cpu_set:
                numas.add(i)
                break
    return numas


def run(worker_id, args, barrier, cpu_affinity_set, results):
    # Set the CPU affinity of this process
    numas = cpus_to_numas(cpu_affinity_set)
    numa.memory.set_interleave_nodes(*numas)
    numa.schedule.run_on_nodes(*numas)

    # Run on given CPUs. The first argument is the process id,
    # and 0 means this present process
    numa.schedule.run_on_cpus(0, *cpu_affinity_set)

    input_shapes = parse_input_shapes(args.input_shapes)

    # Compile the model
    model = compile_model(
        model=args.model_path,
        batch_size=args.batch_size,
        num_cores=len(cpu_affinity_set),
        scheduler="single_stream",
        input_shapes=input_shapes,
    )

    # Cleanly separate compilation and benchmarking
    barrier.wait()

    # Generate random inputs to feed the model
    # TODO(mgoin): should be able to query Engine class instead of loading ONNX
    if input_shapes:
        with override_onnx_input_shapes(args.model_path, input_shapes) as model_path:
            input_list = generate_random_inputs(model_path, args.batch_size)
    else:
        input_list = generate_random_inputs(args.model_path, args.batch_size)

    # Warmup the engine
    singlestream_benchmark(model, input_list, args.warmup_time)

    # Run the singlestream benchmark scenario and collect batch times
    batch_times = singlestream_benchmark(model, input_list, args.time)

    # Cooldown the engine
    singlestream_benchmark(model, input_list, args.warmup_time)

    if len(batch_times) == 0:
        raise Exception(
            "Generated no batch timings, try extending benchmark time with '--time'"
        )

    results[worker_id] = batch_times


def main():
    args = parse_args()

    decide_thread_pinning(args.thread_pinning)

    b = mp.Barrier(args.num_streams)

    # Hardcode affinity sets for elmo
    affinity_sets = [
        {0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20, 21, 22, 23},
        {24, 25, 26, 27, 28, 29, 30, 31},
        {32, 33, 34, 35, 36, 37, 38, 39},
        {40, 41, 42, 43, 44, 45, 46, 47},
        {48, 49, 50, 51, 52, 53, 54, 55},
        {56, 57, 58, 59, 60, 61, 62, 63},
        {64, 65, 66, 67, 68, 69, 70, 71},
        {72, 73, 74, 75, 76, 77, 78, 79},
        {80, 81, 82, 83, 84, 85, 86, 87},
        {88, 89, 90, 91, 92, 93, 94, 95},
        {96, 97, 98, 99, 100, 101, 102, 103},
        {104, 105, 106, 107, 108, 109, 110, 111},
        {112, 113, 114, 115, 116, 117, 118, 119},
        {120, 121, 122, 123, 124, 125, 126, 127},
    ]

    # Make sure we don't try to download the model in parallel
    Zoo.download_model_from_stub(stub=args.model_path)
    orig_model_path = args.model_path
    args.model_path = model_to_path(args.model_path)

    all_batch_times = []
    summed_throughput = 0
    with mp.Manager() as manager:
        results = manager.dict()

        # Generate n-1 workers, and have the original process do its own inferences.
        workers = []
        for i in range(args.num_streams - 1):
            p = mp.Process(target=run, args=(i, args, b, affinity_sets[i], results))
            p.start()
            workers.append(p)
        my_id = args.num_streams - 1
        run(my_id, args, b, affinity_sets[my_id], results)

        for w in workers:
            w.join()

        for i in range(args.num_streams):
            all_batch_times.extend(results[i])

        # Calculate throughput for each stream
        # Note: We want to know all of the executions that could be performed within a
        # given amount of wallclock time. This calculation as-is includes the test overhead
        # such as saving timing results for each iteration so it isn't a best-case but is a
        # realistic case.
        for i in range(args.num_streams):
            first_start_time = min([b[0] for b in results[i]])
            last_end_time = max([b[1] for b in results[i]])
            total_time_executing = last_end_time - first_start_time

            items_per_sec = (args.batch_size * len(results[i])) / total_time_executing
            summed_throughput = summed_throughput + items_per_sec

    # Convert times to milliseconds
    batch_times_ms = [
        (batch_time[1] - batch_time[0]) * 1000 for batch_time in all_batch_times
    ]

    percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    buckets = np.percentile(batch_times_ms, percentiles).tolist()
    percentiles_dict = {
        "{:2.1f}%".format(key): value for key, value in zip(percentiles, buckets)
    }
    benchmark_dict = {
        "items_per_sec": summed_throughput,
        "iterations": len(batch_times_ms),
        "median": np.median(batch_times_ms),
        "mean": np.mean(batch_times_ms),
        "std": np.std(batch_times_ms),
        **percentiles_dict,
    }

    benchmark_result = benchmark_dict

    # Results summary
    print("Original Model Path: {}".format(orig_model_path))
    print("Batch Size: {}".format(args.batch_size))
    print("Throughput (items/sec): {:.4f}".format(benchmark_result["items_per_sec"]))
    print("Latency Mean (ms/batch): {:.4f}".format(benchmark_result["mean"]))
    print("Latency Median (ms/batch): {:.4f}".format(benchmark_result["median"]))
    print("Latency Std (ms/batch): {:.4f}".format(benchmark_result["std"]))
    print("Iterations: {}".format(int(benchmark_result["iterations"])))


if __name__ == "__main__":
    main()
