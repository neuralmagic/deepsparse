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

"""
Benchmarking script for ONNX models with the DeepSparse engine.

##########
Command help:
usage: deepsparse.benchmark [-h] [-b BATCH_SIZE] [-shapes INPUT_SHAPES]
                            [-ncores NUM_CORES] [-s {async,sync,elastic}]
                            [-t TIME] [-w WARMUP_TIME] [-nstreams NUM_STREAMS]
                            [-pin {none,core,numa}]
                            [-e {deepsparse,onnxruntime}] [-q]
                            [-x EXPORT_PATH]
                            model_path

Benchmark ONNX models in the DeepSparse Engine

positional arguments:
  model_path            Path to an ONNX model file or SparseZoo model stub.

optional arguments:
  -h, --help            show this help message and exit.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for. Must be
                        greater than 0.
  -shapes INPUT_SHAPES, --input_shapes INPUT_SHAPES
                        Override the shapes of the inputs, i.e. -shapes
                        "[1,2,3],[4,5,6],[7,8,9]" results in input0=[1,2,3]
                        input1=[4,5,6] input2=[7,8,9].
  -ncores NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system.
  -s {async,sync,elastic}, --scenario {async,sync,elastic}
                        Choose between using the async, sync and elastic
                        scenarios. Sync and async are similar to the single-
                        stream/multi-stream scenarios. Elastic is a newer
                        scenario that behaves similarly to the async scenario
                        but uses a different scheduling backend. Default value
                        is sync.
  -t TIME, --time TIME  The number of seconds the benchmark will run. Default
                        is 10 seconds.
  -w WARMUP_TIME, --warmup_time WARMUP_TIME
                        The number of seconds the benchmark will warmup before
                        running.Default is 2 seconds.
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        The number of streams that will submit inferences in
                        parallel using async scenario. Default is
                        automatically determined for given hardware and may be
                        sub-optimal.
  -pin {none,core,numa}, --thread_pinning {none,core,numa}
                        Enable binding threads to cores ('core' the default),
                        threads to cores on sockets ('numa'), or disable
                        ('none').
  -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                        Inference engine backend to run eval on. Choices are
                        'deepsparse', 'onnxruntime'. Default is 'deepsparse'.
  -q, --quiet           Lower logging verbosity.
  -x EXPORT_PATH, --export_path EXPORT_PATH
                        Store results into a JSON file.

##########
Example on a BERT from SparseZoo:
deepsparse.benchmark \
   zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none

##########
Example on a BERT from SparseZoo with sequence length 512:
deepsparse.benchmark \
   zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none \
   --input_shapes "[1,512],[1,512],[1,512]"

##########
Example on local ONNX model:
deepsparse.benchmark /PATH/TO/model.onnx

##########
Example on local ONNX model at batch size 32 with synchronous (singlestream) execution:
deepsparse.benchmark /PATH/TO/model.onnx --batch_size 32 --scenario sync

"""

import argparse
import importlib
import json
import logging
import os
from typing import Dict

from deepsparse import Scheduler, __version__, compile_model
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


__all__ = ["benchmark_model"]


_LOGGER = logging.getLogger(__name__)

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


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
        "-i",
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
        "-w",
        "--warmup_time",
        type=int,
        default=2,
        help=(
            "The number of seconds the benchmark will warmup before running."
            "Default is 2 seconds."
        ),
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


def decide_thread_pinning(pinning_mode: str) -> None:
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


def parse_scheduler(scenario: str) -> Scheduler:
    scenario = scenario.lower()
    if scenario == "multistream":
        return Scheduler.multi_stream
    elif scenario == "singlestream":
        return Scheduler.single_stream
    elif scenario == "elastic":
        return Scheduler.elastic
    else:
        return Scheduler.multi_stream


def parse_scenario(scenario: str) -> str:
    scenario = scenario.lower()
    if scenario == "async":
        return "multistream"
    elif scenario == "sync":
        return "singlestream"
    elif scenario == "elastic":
        return "elastic"
    else:
        _LOGGER.info(
            "Recieved invalid option for scenario'{}', defaulting to async".format(
                scenario
            )
        )
        return "multistream"


def parse_num_streams(num_streams: int, num_cores: int, scenario: str):
    # If model.num_streams is set, and the scenario is either "multi_stream" or
    # "elastic", use the value of num_streams given to us by the model, otherwise
    # use a semi-sane default value.
    if scenario == "sync" or scenario == "singlestream":
        if num_streams and num_streams > 1:
            _LOGGER.info("num_streams reduced to 1 for singlestream scenario.")
        return 1
    else:
        if num_streams:
            return num_streams
        else:
            default_num_streams = max(1, int(num_cores / 2))
            _LOGGER.info(
                "num_streams default value chosen of {}. "
                "This requires tuning and may be sub-optimal".format(
                    default_num_streams
                )
            )
            return default_num_streams


def load_custom_engine(custom_engine_identifier: str):
    """
    import a custom engine based off the specified `custom_engine_identifier`
    from user specified script

    :param custom_engine_identifier: string in the form of
           '<path_to_the_python_script>:<custom_engine_class_name>
    :return: custom engine class object
    """
    path, engine_object_name = custom_engine_identifier.split(":")
    spec = importlib.util.spec_from_file_location("user_defined_custom_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, engine_object_name)


def benchmark_model(
    model_path: str,
    batch_size: int = 1,
    input_shapes: str = "",
    num_cores: int = None,
    scenario: str = "sync",
    time: int = 10,
    warmup_time: int = 2,
    num_streams: int = None,
    thread_pinning: str = "core",
    engine: str = DEEPSPARSE_ENGINE,
    quiet: bool = False,
    export_path: str = None,
) -> Dict:
    if quiet:
        set_logging_level(logging.WARN)

    decide_thread_pinning(thread_pinning)

    scenario = parse_scenario(scenario.lower())
    scheduler = parse_scheduler(scenario)
    input_shapes = parse_input_shapes(input_shapes)

    orig_model_path = model_path
    model_path = model_to_path(model_path)
    num_streams = parse_num_streams(num_streams, num_cores, scenario)

    # Compile the ONNX into a runnable model
    if engine == DEEPSPARSE_ENGINE:
        model = compile_model(
            model=model_path,
            batch_size=batch_size,
            num_cores=num_cores,
            num_streams=num_streams,
            scheduler=scheduler,
            input_shapes=input_shapes,
        )
    elif engine == ORT_ENGINE:
        model = ORTEngine(
            model=model_path,
            batch_size=batch_size,
            num_cores=num_cores,
            input_shapes=input_shapes,
        )
    elif ":" in engine:
        engine = load_custom_engine(custom_engine_identifier=engine)
        model = engine(
            model_path=model_path,
            batch_size=batch_size,
            num_cores=num_cores,
        )
    else:
        raise ValueError(f"Invalid engine choice '{engine}'")
    _LOGGER.info(model)

    # Generate random inputs to feed the model
    # TODO(mgoin): should be able to query Engine class instead of loading ONNX
    if input_shapes:
        with override_onnx_input_shapes(model_path, input_shapes) as model_path:
            input_list = generate_random_inputs(model_path, batch_size)
    elif hasattr(engine, "generate_random_inputs"):
        input_list = engine.generate_random_inputs(batch_size=batch_size)
    else:
        input_list = generate_random_inputs(model_path, batch_size)

    # Benchmark
    _LOGGER.info(
        "Starting '{}' performance measurements for {} seconds".format(scenario, time)
    )
    benchmark_result = model_stream_benchmark(
        model,
        input_list,
        scenario=scenario,
        seconds_to_run=time,
        seconds_to_warmup=warmup_time,
        num_streams=num_streams,
    )

    export_dict = {
        "engine": str(model),
        "version": __version__,
        "orig_model_path": orig_model_path,
        "model_path": model_path,
        "batch_size": batch_size,
        "input_shapes": input_shapes,
        "num_cores": num_cores,
        "scenario": scenario,
        "scheduler": str(model.scheduler),
        "seconds_to_run": time,
        "num_streams": num_streams,
        "benchmark_result": benchmark_result,
    }

    # Export results
    if export_path:
        _LOGGER.info("Saving benchmark results to JSON file at {}".format(export_path))
        with open(export_path, "w") as out:
            json.dump(export_dict, out, indent=2)

    return export_dict


def main():

    args = parse_args()

    result = benchmark_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        input_shapes=args.input_shapes,
        num_cores=args.num_cores,
        scenario=args.scenario,
        time=args.time,
        warmup_time=args.warmup_time,
        num_streams=args.num_streams,
        thread_pinning=args.thread_pinning,
        engine=args.engine,
        quiet=args.quiet,
        export_path=args.export_path,
    )

    # Results summary
    print("Original Model Path: {}".format(args.model_path))
    print("Batch Size: {}".format(args.batch_size))
    print("Scenario: {}".format(args.scenario))
    print(
        "Throughput (items/sec): {:.4f}".format(
            result["benchmark_result"]["items_per_sec"]
        )
    )
    print("Latency Mean (ms/batch): {:.4f}".format(result["benchmark_result"]["mean"]))
    print(
        "Latency Median (ms/batch): {:.4f}".format(result["benchmark_result"]["median"])
    )
    print("Latency Std (ms/batch): {:.4f}".format(result["benchmark_result"]["std"]))
    print("Iterations: {}".format(int(result["benchmark_result"]["iterations"])))


if __name__ == "__main__":
    main()
