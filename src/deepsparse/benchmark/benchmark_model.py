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
usage: deepsparse.benchmark [-h] [-b BATCH_SIZE] [-i INPUT_SHAPES]
                            [-ncores NUM_CORES] [-s {async,sync,elastic}]
                            [-t TIME] [-w WARMUP_TIME] [-nstreams NUM_STREAMS]
                            [-seq_len SEQUENCE_LENGTH]
                            [-input_ids_len INPUT_IDS_LENGTH]
                            [-pin {none,core,numa}] [-e ENGINE]
                            [--no-internal-kv-cache] [-q] [-x EXPORT_PATH]
                            [--disable-kv-cache-overrides]
                            model_path

Benchmark ONNX models in the DeepSparse Engine

positional arguments:
  model_path            Path to an ONNX model file or SparseZoo model stub

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for. Must be
                        greater than 0
  -i INPUT_SHAPES, -shapes INPUT_SHAPES, --input_shapes INPUT_SHAPES
                        Override the shapes of the inputs, i.e. -shapes
                        "[1,2,3],[4,5,6],[7,8,9]" results in input0=[1,2,3]
                        input1=[4,5,6] input2=[7,8,9]
  -ncores NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system
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
  -seq_len SEQUENCE_LENGTH, --sequence_length SEQUENCE_LENGTH
                        The sequence length to run the KV cache supported
                        model benchmarks for. Must be seq_len >= 1, default is
                        None
  -input_ids_len INPUT_IDS_LENGTH, --input_ids_length INPUT_IDS_LENGTH
                        The input ids length to run the KV cache supported
                        model benchmarks for. Must be 1 <= input_ids_len <=
                        seq_len, default is 1
  -pin {none,core,numa}, --thread_pinning {none,core,numa}
                        Enable binding threads to cores ('core' the default),
                        threads to cores on sockets ('numa'), or disable
                        ('none')
  -e ENGINE, --engine ENGINE
                        Inference engine backend to run eval on. Choices are
                        'deepsparse', 'onnxruntime'. Default is 'deepsparse'.
                        Can also specify a user defined engine class by giving
                        the script and class name in the following format
                        <path to python script>:<Engine Class name>. This
                        engine class will be dynamically imported during
                        runtime
  --no-internal-kv-cache, --no_internal_kv_cache
                        DeepSparse engine only - If not present, and model has
                        KV cache, KV Cache state will be managed within the
                        compiled deepsparse model. This is preferred when
                        applicable for best performance. Set flag to disable
  -q, --quiet           Lower logging verbosity
  -x EXPORT_PATH, --export_path EXPORT_PATH
                        Store results into a JSON file
  --disable-kv-cache-overrides, --disable_kv_cache_overrides
                        If set, it will not alter the model
                        with kv cache overrides

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
Example on a CodeGen (model with KV cache support)
from SparseZoo with input_ids_length 10 and sequence length 256:
deepsparse.benchmark \
   zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/pruned50-none \
   --input_ids_length 10 --sequence_length 256

##########
Example on local ONNX model:
deepsparse.benchmark /PATH/TO/model.onnx

##########
Example on local ONNX model at batch size 32 with synchronous (singlestream) execution:
deepsparse.benchmark /PATH/TO/model.onnx --batch_size 32 --scenario sync

"""  # noqa E501

import argparse
import importlib
import json
import logging
from typing import Dict, Optional

from deepsparse import Engine, __version__
from deepsparse.benchmark.helpers import (
    decide_thread_pinning,
    parse_num_streams,
    parse_scenario,
    parse_scheduler,
)
from deepsparse.benchmark.ort_engine import ORTEngine
from deepsparse.benchmark.stream_benchmark import model_stream_benchmark
from deepsparse.cpu import cpu_architecture
from deepsparse.log import set_logging_level
from deepsparse.utils import (
    generate_random_inputs,
    has_model_kv_cache,
    infer_sequence_length,
    model_to_path,
    override_onnx_input_shapes,
    overwrite_onnx_model_inputs_for_kv_cache_models,
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
        "-seq_len",
        "--sequence_length",
        type=int,
        default=None,
        help="The sequence length to run the KV cache supported model "
        "benchmarks for. Must be seq_len >= 1, default is None",
    )
    parser.add_argument(
        "-input_ids_len",
        "--input_ids_length",
        type=int,
        default=1,
        help="The input ids length to run the KV cache supported model "
        "benchmarks for. Must be 1 <= input_ids_len <= seq_len, default is 1",
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
        "--no-internal-kv-cache",
        "--no_internal_kv_cache",
        help=(
            "DeepSparse engine only - If not present, and model has KV cache, "
            "KV Cache state will be managed within the compiled deepsparse "
            "model. This is preferred when applicable for best performance. Set "
            "flag to disable"
        ),
        action="store_true",
        default=False,
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
    parser.add_argument(
        "--disable-kv-cache-overrides",
        "--disable_kv_cache_overrides",
        help=("If set, it will not alter the model with kv cache overrides"),
        action="store_true",
        default=False,
    )

    return parser.parse_args()


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
    num_cores: Optional[int] = None,
    scenario: str = "sync",
    time: int = 10,
    warmup_time: int = 2,
    num_streams: Optional[int] = None,
    sequence_length: Optional[int] = None,
    input_ids_length: Optional[int] = 1,
    thread_pinning: str = "core",
    engine: str = DEEPSPARSE_ENGINE,
    internal_kv_cache: bool = False,
    quiet: bool = False,
    export_path: Optional[str] = None,
    disable_kv_cache_overrides: bool = False,
) -> Dict:
    if quiet:
        set_logging_level(logging.WARN)

    if num_cores is None:
        num_cores = cpu_architecture().num_available_physical_cores

    decide_thread_pinning(thread_pinning)

    scenario = parse_scenario(scenario.lower())
    scheduler = parse_scheduler(scenario)
    input_shapes = parse_input_shapes(input_shapes)

    orig_model_path = model_path
    model_path = model_to_path(model_path)

    cached_outputs = None
    if has_model_kv_cache(model_path):
        if not disable_kv_cache_overrides:
            if not sequence_length:
                sequence_length = infer_sequence_length(model_path)
            if input_ids_length > sequence_length:
                raise ValueError(
                    f"input_ids_length: {input_ids_length} "
                    f"must be less than sequence_length: {sequence_length}"
                )

            _LOGGER.info(
                "Found model with KV cache support. "
                "Benchmarking the autoregressive model with "
                f"input_ids_length: {input_ids_length} and "
                f"sequence length: {sequence_length}."
            )

            (
                model_path,
                cached_outputs,
                _,
            ) = overwrite_onnx_model_inputs_for_kv_cache_models(
                onnx_file_path=model_path,
                input_ids_length=input_ids_length,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )

        if internal_kv_cache and engine != DEEPSPARSE_ENGINE:
            raise ValueError(
                "Attempting to benchmark a model using engine: "
                f"{engine} and internal_kv_cache set to True. "
                "The use of internal_kv_cache is only "
                f"supported for the engine: {DEEPSPARSE_ENGINE}. "
                f"To disable the use of the internal_kv_cache, "
                f"set the flag: --no-internal-kv-cache"
            )

        _LOGGER.info(
            f"Benchmarking Engine: {engine} with "
            f"{'internal' if internal_kv_cache else 'external'} KV cache management"
        )
    else:
        input_ids_length = None
        sequence_length = None
        internal_kv_cache = False

    num_streams = parse_num_streams(num_streams, num_cores, scenario)

    # Compile the ONNX into a runnable model
    if engine == DEEPSPARSE_ENGINE:
        model = Engine(
            model=model_path,
            batch_size=batch_size,
            num_cores=num_cores,
            num_streams=num_streams,
            scheduler=scheduler,
            input_shapes=input_shapes,
            cached_outputs=cached_outputs if internal_kv_cache else None,
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
    elif hasattr(model, "generate_random_inputs"):
        input_list = model.generate_random_inputs()
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
        internal_kv_cache=internal_kv_cache,
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
        "fraction_of_supported_ops": getattr(model, "fraction_of_supported_ops", None),
    }
    if sequence_length and input_ids_length:
        export_dict["sequence_length"] = sequence_length
        export_dict["input_ids_length"] = input_ids_length

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
        sequence_length=args.sequence_length,
        input_ids_length=args.input_ids_length,
        thread_pinning=args.thread_pinning,
        engine=args.engine,
        internal_kv_cache=not args.no_internal_kv_cache,
        quiet=args.quiet,
        export_path=args.export_path,
        disable_kv_cache_overrides=args.disable_kv_cache_overrides,
    )

    # Results summary
    print("Original Model Path: {}".format(args.model_path))
    print("Batch Size: {}".format(args.batch_size))
    if args.sequence_length:
        print("Sequence Length: {}".format(args.sequence_length))
        print("Input IDs Length: {}".format(args.input_ids_length))
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
