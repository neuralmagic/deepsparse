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
import logging
import os

from deepsparse import Scheduler, compile_model
from deepsparse.benchmark_model.stream_benchmark import model_stream_benchmark
from deepsparse.utils import (
    generate_random_inputs,
    model_to_path,
    override_onnx_input_shapes,
    parse_input_shapes,
)
from deepsparse.utils.log import log_init


log = log_init(os.path.basename(__file__))


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
        "-ncores",
        "--num_cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the analysis on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        default="async",
        choices=["async", "sync"],
        help=(
            "Choose between using a sync/async scenarios. This is similar to the "
            "single-stream/multi-stream scenarios. Default value is async."
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


def decide_thread_pinning(pinning_mode: str):
    pinning_mode = pinning_mode.lower()

    if pinning_mode in "core":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "1"
        log.info("Thread pinning to cores enabled")
    elif pinning_mode in "numa":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "1"
        log.info("Thread pinning to socket/numa nodes enabled")
    elif pinning_mode in "none":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "0"
        log.info("Thread pinning disabled, performance may be sub-optimal")
    else:
        log.info(
            "Recieved invalid option for thread_pinning '{}', skipping".format(
                pinning_mode
            )
        )


def main():

    args = parse_args()

    if args.quiet:
        log.setLevel(logging.WARN)

    decide_thread_pinning(args.thread_pinning)

    scheduler = (
        Scheduler.multi_stream
        if args.scenario.lower() == "async"
        else Scheduler.single_stream
    )
    scenario = "multistream" if scheduler is Scheduler.multi_stream else "singlestream"
    input_shapes = parse_input_shapes(args.input_shapes)

    args.model_path = model_to_path(args.model_path)

    # Compile the ONNX into the DeepSparse Engine
    model = compile_model(
        model=args.model_path,
        batch_size=args.batch_size,
        num_cores=args.num_cores,
        scheduler=scheduler,
        input_shapes=input_shapes,
    )
    log.info(model)

    # Generate random inputs to feed the model
    # TODO(mgoin): should be able to query Engine class instead of loading ONNX
    if input_shapes:
        with override_onnx_input_shapes(args.model_path, input_shapes) as model_path:
            input_list = generate_random_inputs(model_path, args.batch_size)
    else:
        input_list = generate_random_inputs(args.model_path, args.batch_size)

    # If num_streams isn't defined, find a default
    if not args.num_streams and scenario not in "singlestream":
        args.num_streams = int(model.num_cores / 2)
        log.info(
            "num_streams default value chosen of {}. "
            "This requires tuning and may be sub-optimal".format(args.num_streams)
        )

    # Benchmark
    log.info(
        "Starting '{}' performance measurements for {} seconds".format(
            args.scenario, args.time
        )
    )
    benchmark_result = model_stream_benchmark(
        model,
        input_list,
        scenario,
        args.time,
        args.num_streams,
    )

    # Results summary
    headers = [
        "Scenario",
        "Throughput",
        "Mean (ms)",
        "Median (ms)",
        "Std (ms)",
        "Iterations",
    ]
    header_format = "| {:<13} " * len(headers)
    print(header_format.format(*headers))
    row_format = "| {:<13} " + "| {:<13.4f} " * (len(headers) - 1)
    print(
        row_format.format(
            scenario,
            benchmark_result["items_per_sec"],
            benchmark_result["mean"],
            benchmark_result["median"],
            benchmark_result["std"],
            int(benchmark_result["iterations"]),
        )
    )

    # Export results
    if args.export_path:
        export_dict = {
            "engine": model,
            "onnx_filename": args.model_path,
            "batch_size": args.batch_size,
            "num_cores": args.num_cores,
            "scheduler": model.scheduler,
            "seconds_to_run": args.time,
            "num_streams": args.num_streams,
            "benchmark_result": benchmark_result,
        }
        print(f"Saving JSON output to {args.export_path}")
        print(export_dict)

        with open(args.export_path, "w") as out:
            json.dump(export_dict, out, indent=2)


if __name__ == "__main__":
    main()
