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
from typing import Dict

from deepsparse import __version__, compile_model
from deepsparse import Pipeline
from deepsparse.benchmark.ort_engine import ORTEngine
from deepsparse.benchmark.stream_benchmark import model_stream_benchmark
from deepsparse.cpu import cpu_architecture
from deepsparse.log import set_logging_level
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
    time: int = 10,
    num_streams: int = None,
    thread_pinning: str = "core",
    engine: str = DEEPSPARSE_ENGINE,
    quiet: bool = False,
    export_path: str = None,
) -> Dict:
    
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

    data_length = config['length']
    num_examples = config['num_examples']
    examples = []
    if config['input_data_type'] == "string":
        for _ in range(num_examples):
            rand_string = ''.join(random.choices(string.printable, k=data_length))
            examples.append(rand_string)
    print(examples)

    pipeline = Pipeline.create(task=task, model_path=model_path)
    output = pipeline(examples)
    print(output)

    return {}


def main():
    args = parse_args()

    result = benchmark_pipeline(
        model_path=args.model_path,
        task=args.task_name,
        input_config = args.input_config,
        input_type = args.input_type,
        batch_size=args.batch_size,
        num_cores=args.num_cores,
        scenario=args.scenario,
        time=args.time,
        num_streams=args.num_streams,
        thread_pinning=args.thread_pinning,
        engine=args.engine,
        quiet=args.quiet,
        export_path=args.export_path,
    )

    # Results summary
    print("Original Model Path: {}".format(args.model_path))
    print("Task: {}".format(args.task_name))
    print("Input Type: {}".format(args.input_type))
    print("Batch Size: {}".format(args.batch_size))
    print("Scenario: {}".format(args.scenario))


if __name__ == "__main__":
    main()
