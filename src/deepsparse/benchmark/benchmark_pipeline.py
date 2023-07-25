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
import importlib
import json
import string
import logging
import random
import os
from typing import Dict, List
import time
import numpy

from deepsparse import __version__, compile_model
from deepsparse import Pipeline
from deepsparse.benchmark.ort_engine import ORTEngine
from deepsparse.benchmark.stream_benchmark import model_stream_benchmark
from deepsparse.cpu import cpu_architecture
from deepsparse.log import set_logging_level
from deepsparse.utils.timer import StagedTimer
from deepsparse.utils import (
    generate_random_inputs,
    model_to_path,
    override_onnx_input_shapes,
    parse_input_shapes,
)
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
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        help=(
            "Inference engine backend to run eval on. Choices are 'deepsparse', "
            "'onnxruntime'. Default is 'deepsparse'. Can also specify a user "
            "defined engine class by giving the script and class name in the "
            "following format <path to python script>:<Engine Class name>. This "
            "engine class will be dynamically imported during runtime"
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



def parse_input_config(input_config_file: str) -> Dict[str, object]:
    config_file = open(input_config_file)
    config = json.load(config_file)
    config_file.close()
    return config

def benchmark_pipeline(
    model_path: str,
    task: str,
    input_config: str,
    input_type: str = "dummy",
    batch_size: int = 1,
    num_cores: int = None,
    scenario: str = "sync",
    seconds_to_run: int = 10,
    num_streams: int = None,
    thread_pinning: str = "core",
    engine: str = DEEPSPARSE_ENGINE,
    quiet: bool = False,
    export_path: str = None,
) -> List[StagedTimer]:
    
    if quiet:
        set_logging_level(logging.WARN)

    if num_cores is None:
        num_cores = cpu_architecture().num_available_physical_cores

    decide_thread_pinning(thread_pinning, _LOGGER)
    scenario = parse_scenario(scenario.lower(), _LOGGER)
    scheduler = parse_scheduler(scenario)
    num_streams = parse_num_streams(num_streams, num_cores, scenario, _LOGGER)

    # Compile the ONNX into a runnable model
    if engine == DEEPSPARSE_ENGINE:
        model = compile_model(
            model=model_path,
            batch_size=batch_size,
            num_cores=num_cores,
            num_streams=num_streams,
            scheduler=scheduler,
        )
    elif engine == ORT_ENGINE:
        model = ORTEngine(
            model=model_path,
            batch_size=batch_size,
            num_cores=num_cores,
        )
    else:
        raise ValueError(f"Invalid engine choice '{engine}'")
    _LOGGER.info(model)
    
    config = parse_input_config(input_config)
    pipeline = Pipeline.create(task=task, model_path=model_path)

    input_data = []
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

    benchmark_end_time = time.perf_counter() + seconds_to_run
    batch_timings = []
    while time.perf_counter() < benchmark_end_time:
        output = pipeline(inputs)
        batch_timings.append(pipeline.timer_manager.latest)

    return batch_timings


def main():
    args = parse_args()

    print("Original Model Path: {}".format(args.model_path))
    print("Task: {}".format(args.task_name))
    print("Input Type: {}".format(args.input_type))
    print("Batch Size: {}".format(args.batch_size))
    print("Scenario: {}".format(args.scenario))

    result = benchmark_pipeline(
        model_path=args.model_path,
        task=args.task_name,
        input_config = args.input_config,
        input_type = args.input_type,
        batch_size=args.batch_size,
        num_cores=args.num_cores,
        scenario=args.scenario,
        seconds_to_run=args.time,
        num_streams=args.num_streams,
        thread_pinning=args.thread_pinning,
        engine=args.engine,
        quiet=args.quiet,
        export_path=args.export_path,
    )

    # Results summary
    batches_processed = len(result)
    total_time = sum(st.times['total_inference'] for st in result)
    print("Processed {} batches in {} seconds".format(batches_processed, total_time))
    throughput = round(batches_processed / total_time, 4)
    print("Throughput: {} batches/sec".format(throughput))
    total_pre_process = sum(st.times['pre_process'] for st in result)
    total_post_process = sum(st.times['post_process'] for st in result)
    total_engine_forward = sum(st.times['engine_forward'] for st in result)

    avg_pre_process = round(total_pre_process / batches_processed * 1000, 4)
    avg_post_process = round(total_post_process / batches_processed * 1000, 4)
    avg_engine_forward = round(total_engine_forward / batches_processed * 1000, 4)

    print("Average Pre-Process: {} ms".format(avg_pre_process))
    print("Average Post-Process: {} ms".format(avg_post_process))
    print("Average Engine Forward: {} ms".format(avg_engine_forward))

    total_time = total_pre_process + total_post_process + total_engine_forward
    percent_pre = round(total_pre_process / total_time * 100, 2)
    percent_post = round(total_post_process / total_time * 100, 2)
    percent_forward = round(total_engine_forward / total_time * 100, 2)
    print("{}% Pre-processing, {}% Post-processing, {}% Inference".format(percent_pre, percent_post, percent_forward))


if __name__ == "__main__":
    main()
