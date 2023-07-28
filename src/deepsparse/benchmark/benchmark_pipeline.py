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
Benchmark DeepSparse Pipelines

##########
Command help:
usage: deepsparse.benchmark_pipeline [-h] [-c INPUT_CONFIG] [-b BATCH_SIZE]
                                     [-ncores NUM_CORES] [-s {async,sync,elastic}]
                                     [-t TIME] [-w WARMUP_TIME] [-nstreams NUM_STREAMS]
                                     [-pin {none,core,numa}] [-e ENGINE]
                                     [-q] [-x EXPORT_PATH] task_name model_path

positional arguments:
  task_name             Type of pipeline to run
  model_path            Path to an ONNX model file or SparseZoo model stub

optional arguments:
  -h, --help            show this help message and exit
  -c INPUT_CONFIG, --input_config INPUT_CONFIG
                        JSON file containing schema for input data
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for. Must be greater than 0
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
Example ResNet image classification for 30 seconds with a batch size of 32:
```
deepsparse.benchmark_pipeline \
    image_classification \
    zoo:cv/classification/resnet_v1-50_2x/pytorch/sparseml/imagenet/base-none \
    -c config.json -t 60 -b 32

##########
Example CodeGen text generation for 30 seconds asynchronously
deepsparse.benchmark_pipeline \
    text_generation \
    zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/
    bigpython_bigquery_thepile/pruned50-none \
    -c config.json -t 30 -s async
"""

import argparse
import json
import logging
import queue
import threading
import time
from typing import Dict, List, Tuple

import numpy

from deepsparse import Pipeline, __version__
from deepsparse.benchmark.data_creation import (
    SchemaType,
    generate_image_data,
    generate_question_data,
    generate_text_data,
    get_input_schema_type,
    load_image_data,
    load_question_data,
    load_text_data,
)
from deepsparse.benchmark.helpers import (
    decide_thread_pinning,
    parse_input_config,
    parse_num_streams,
    parse_scenario,
    parse_scheduler,
)
from deepsparse.cpu import cpu_architecture
from deepsparse.log import set_logging_level
from deepsparse.utils.timer import StagedTimer


__all__ = ["benchmark_pipeline"]


_LOGGER = logging.getLogger(__name__)

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

DUMMY_INPUT_TYPE = "dummy"
REAL_INPUT_TYPE = "real"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DeepSparse Pipelines")
    parser.add_argument("task_name", type=str, help="Type of pipeline to run")
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
        help="JSON file containing schema for input data",
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


class PipelineExecutorThread(threading.Thread):
    def __init__(
        self,
        pipeline: Pipeline,
        inputs: List[any],
        time_queue: queue.Queue,
        max_time: float,
    ):
        super(PipelineExecutorThread, self).__init__()
        self._pipeline = pipeline
        self._inputs = inputs
        self._time_queue = time_queue
        self._max_time = max_time

    def run(self):
        while time.perf_counter() < self._max_time:
            _ = self._pipeline(self._inputs)
            self._time_queue.put(self._pipeline.timer_manager.latest)


def singlestream_benchmark(
    pipeline: Pipeline, inputs: List[any], seconds_to_run: float
) -> List[StagedTimer]:
    benchmark_end_time = time.perf_counter() + seconds_to_run
    batch_timings = []
    while time.perf_counter() < benchmark_end_time:
        _ = pipeline(inputs)
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

    for thread in range(num_streams):
        threads.append(PipelineExecutorThread(pipeline, inputs, time_queue, max_time))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return list(time_queue.queue)


def create_input_schema(
    pipeline: Pipeline, input_type: str, batch_size: int, config: Dict
) -> any:
    input_schema_requirement = get_input_schema_type(pipeline)
    kwargs = {}
    if "input_schema_kwargs" in config:
        kwargs = config["input_schema_kwargs"]

    if input_type == DUMMY_INPUT_TYPE:
        if input_schema_requirement == SchemaType.IMAGE:
            input_data = generate_image_data(config, batch_size)
            inputs = pipeline.input_schema(images=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_SEQ:
            input_data = generate_text_data(config, batch_size)
            inputs = pipeline.input_schema(sequences=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_INPUT:
            input_data = generate_text_data(config, batch_size)
            inputs = pipeline.input_schema(inputs=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.QUESTION:
            _LOGGER.warn(
                "Only batch size of 1 supported for Question Answering Pipeline"
            )
            question, context = generate_question_data(config)
            inputs = pipeline.input_schema(question=question, context=context, **kwargs)
    elif input_type == REAL_INPUT_TYPE:
        if input_schema_requirement == SchemaType.IMAGE:
            input_data = load_image_data(config, batch_size)
            inputs = pipeline.input_schema(images=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_SEQ:
            input_data = load_text_data(config)
            inputs = pipeline.input_schema(sequences=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_INPUT:
            input_data = load_text_data(config, batch_size)
            inputs = pipeline.input_schema(inputs=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.QUESTION:
            _LOGGER.warn(
                "Only batch size of 1 supported for Question Answering Pipeline"
            )
            question, context = load_question_data(config)
            inputs = pipeline.input_schema(question=question, context=context, **kwargs)
    else:
        raise Exception(f"Unknown input type '{input_type}'")

    return inputs


def benchmark_pipeline(
    model_path: str,
    task: str,
    config: Dict,
    batch_size: int = 1,
    num_cores: int = None,
    scenario: str = "sync",
    seconds_to_run: int = 10,
    warmup_time: int = 2,
    num_streams: int = None,
    thread_pinning: str = "core",
    engine: str = DEEPSPARSE_ENGINE,
    quiet: bool = False,
) -> Tuple[List[StagedTimer], float]:

    if quiet:
        set_logging_level(logging.WARN)

    if num_cores is None:
        num_cores = cpu_architecture().num_available_physical_cores

    decide_thread_pinning(thread_pinning)
    scenario = parse_scenario(scenario.lower())
    scheduler = parse_scheduler(scenario)
    num_streams = parse_num_streams(num_streams, num_cores, scenario)

    input_type = config["data_type"]
    kwargs = {}
    if "pipeline_kwargs" in config:
        kwargs = config["pipeline_kwargs"]
    pipeline = Pipeline.create(
        task=task,
        model_path=model_path,
        engine_type=engine,
        scheduler=scheduler,
        num_cores=num_cores,
        **kwargs,
    )
    inputs = create_input_schema(pipeline, input_type, batch_size, config)

    if scenario == "singlestream":
        singlestream_benchmark(pipeline, inputs, warmup_time)
        start_time = time.perf_counter()
        batch_times = singlestream_benchmark(pipeline, inputs, seconds_to_run)
    elif scenario == "multistream":
        multistream_benchmark(pipeline, inputs, warmup_time, num_streams)
        start_time = time.perf_counter()
        batch_times = multistream_benchmark(
            pipeline, inputs, seconds_to_run, num_streams
        )
    elif scenario == "elastic":
        multistream_benchmark(pipeline, inputs, warmup_time, num_streams)
        start_time = time.perf_counter()
        batch_times = multistream_benchmark(
            pipeline, inputs, seconds_to_run, num_streams
        )
    else:
        raise Exception(f"Unknown scenario '{scenario}'")

    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    if len(batch_times) == 0:
        raise Exception(
            "Generated no batch timings, try extending benchmark time with '--time'"
        )

    return batch_times, total_run_time


def calculate_statistics(batch_times_ms: List[float], total_run_time_ms: float) -> Dict:
    percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    buckets = numpy.percentile(batch_times_ms, percentiles).tolist()
    percentiles_dict = {
        "{:2.1f}%".format(key): value for key, value in zip(percentiles, buckets)
    }

    benchmark_dict = {
        "total_percentage": sum(batch_times_ms) / total_run_time_ms * 100,
        "median": numpy.median(batch_times_ms),
        "mean": numpy.mean(batch_times_ms),
        "std": numpy.std(batch_times_ms),
        **percentiles_dict,
    }
    return benchmark_dict


def calculate_section_stats(
    batch_times: List[StagedTimer], total_run_time: float
) -> Dict[str, Dict]:
    compute_sections = batch_times[0].stages
    total_run_time_ms = total_run_time * 1000

    sections = {}
    for section in compute_sections:
        section_times = [st.times[section] * 1000 for st in batch_times]
        sections[section] = calculate_statistics(section_times, total_run_time_ms)

    return sections


def main():
    args = parse_args()
    config = parse_input_config(args.input_config)

    print("Original Model Path: {}".format(args.model_path))
    print("Task: {}".format(args.task_name))
    print("Batch Size: {}".format(args.batch_size))
    print("Scenario: {}".format(args.scenario))

    batch_times, total_run_time = benchmark_pipeline(
        model_path=args.model_path,
        task=args.task_name,
        config=config,
        batch_size=args.batch_size,
        num_cores=args.num_cores,
        scenario=args.scenario,
        seconds_to_run=args.time,
        warmup_time=args.warmup_time,
        num_streams=args.num_streams,
        thread_pinning=args.thread_pinning,
        engine=args.engine,
        quiet=args.quiet,
    )

    section_stats = calculate_section_stats(batch_times, total_run_time)
    items_per_sec = (len(batch_times) * args.batch_size) / total_run_time

    benchmark_results = {
        "items_per_sec": items_per_sec,
        "seconds_ran": total_run_time,
        "iterations": len(batch_times),
        "compute_sections": section_stats,
    }

    export_dict = {
        "engine": args.engine,
        "version": __version__,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "num_cores": args.num_cores,
        "scenario": args.scenario,
        "seconds_to_run": time,
        "num_streams": args.num_streams,
        "input_config": config,
        "benchmark_results": benchmark_results,
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
    print("Iterations: {}".format(int(benchmark_results["iterations"])))
    print("Total Runtime: {:.4f}".format(total_run_time))
    print("Throughput (items/sec): {:.4f}".format(benchmark_results["items_per_sec"]))

    print("Processing Time Breakdown: ")
    compute_sections = batch_times[0].stages
    for section in compute_sections:
        print(
            "     {}: {:.2f}%".format(
                section, section_stats[section]["total_percentage"]
            )
        )

    print("Mean Latency Breakdown (ms/batch): ")
    for section in compute_sections:
        print("     {}: {:.4f}".format(section, section_stats[section]["mean"]))


if __name__ == "__main__":
    main()
