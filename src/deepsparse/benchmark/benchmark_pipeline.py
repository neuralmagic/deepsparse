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

Refer to README for config.json examples
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import click
import numpy
from pydantic import BaseModel

from deepsparse import Pipeline, __version__
from deepsparse.benchmark.config import PipelineBenchmarkConfig, PipelineInputType
from deepsparse.benchmark.data_creation import (
    SchemaType,
    generate_random_image_data,
    generate_random_question_data,
    generate_random_text_data,
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


__all__ = ["benchmark_pipeline"]


_LOGGER = logging.getLogger(__name__)

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


class PipelineExecutorThread(threading.Thread):
    """
    Run pipeline repeatedly on inputs for max_time seconds, storing the runtime of each
    section of the pipeline in its timer manager

    For intended usage, see multistream_benchmark
    """

    def __init__(
        self,
        pipeline: Pipeline,
        inputs: List[any],
        max_time: float,
    ):
        super(PipelineExecutorThread, self).__init__()
        self._pipeline = pipeline
        self._inputs = inputs
        self._max_time = max_time

    def run(self):
        while time.perf_counter() < self._max_time:
            _ = self._pipeline(self._inputs)


def singlestream_benchmark(
    pipeline: Pipeline, inputs: List[any], seconds_to_run: float
):
    """
    Run pipeline repeatedly on inputs for max_time seconds, storing the runtime of each
    section of the pipeline in its timer manager

    :param pipeline: pipeline to execute
    :param inputs: inputs to pass through pipeline
    :param seconds_to_run: how long to run pipeline for
    """
    benchmark_end_time = time.perf_counter() + seconds_to_run
    while time.perf_counter() < benchmark_end_time:
        _ = pipeline(inputs)


def multistream_benchmark(
    pipeline: Pipeline,
    inputs: List[any],
    seconds_to_run: float,
    num_streams: int,
):
    """
    Create num_streams threads, each of which calls PipelineExecutorThread.run() for
    seconds_to_run seconds. All timing info stored in pipeline.timer_manager

    :param pipeline: pipeline to execute
    :param inputs: inputs to pass through pipeline
    :param seconds_to_run: how long to run pipeline for
    :param num_streams: number of threads to launch
    """
    max_time = time.perf_counter() + seconds_to_run
    threads = []

    for thread in range(num_streams):
        threads.append(PipelineExecutorThread(pipeline, inputs, max_time))

    for thread in threads:
        thread.start()  # triggers PipelineExecutorThread.run()

    for thread in threads:
        thread.join()


def create_input_schema(
    pipeline: Pipeline,
    input_type: PipelineInputType,
    batch_size: int,
    config: PipelineBenchmarkConfig,
) -> BaseModel:
    input_schema_requirement = get_input_schema_type(pipeline)
    kwargs = config.input_schema_kwargs

    if input_type == PipelineInputType.DUMMY:
        if input_schema_requirement == SchemaType.IMAGE:
            input_data = generate_random_image_data(config, batch_size)
            inputs = pipeline.input_schema(images=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_SEQ:
            input_data = generate_random_text_data(config, batch_size)
            inputs = pipeline.input_schema(sequences=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_INPUT:
            input_data = generate_random_text_data(config, batch_size)
            inputs = pipeline.input_schema(inputs=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.QUESTION:
            question, context = generate_random_question_data(config, batch_size)
            inputs = pipeline.input_schema(question=question, context=context, **kwargs)
    elif input_type == PipelineInputType.REAL:
        if input_schema_requirement == SchemaType.IMAGE:
            input_data = load_image_data(config, batch_size)
            inputs = pipeline.input_schema(images=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_SEQ:
            input_data = load_text_data(config, batch_size)
            inputs = pipeline.input_schema(sequences=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.TEXT_INPUT:
            input_data = load_text_data(config, batch_size)
            inputs = pipeline.input_schema(inputs=input_data, **kwargs)
        elif input_schema_requirement == SchemaType.QUESTION:
            question, context = load_question_data(config, batch_size)
            inputs = pipeline.input_schema(question=question, context=context, **kwargs)
    else:
        raise Exception(f"Unknown input type '{input_type}'")

    return inputs


def benchmark_pipeline(
    model_path: str,
    task: str,
    config: Optional[PipelineBenchmarkConfig] = None,
    batch_size: int = 1,
    num_cores: int = None,
    scenario: str = "sync",
    seconds_to_run: int = 10,
    warmup_time: int = 2,
    num_streams: int = None,
    thread_pinning: str = "core",
    engine: str = DEEPSPARSE_ENGINE,
    quiet: bool = False,
) -> Tuple[Dict[str, List[float]], float]:
    """
    Run a benchmark over the specified pipeline, tracking timings for pre-processing,
    forward pass and post-processing. Results are printed to the console and optionally
    exported to a json file.

    :param model_path: path to onnx model
    :param task: name of pipeline to run
    :param config: configuration for pipeline inputs
    :param batch_size: number of inputs to process each forward pass
    :param num_cores: number of physical cores to run on
    :param scenario: sync, async or elastic processing
    :param seconds_to_run: number of seconds to run benchmark for
    :param warmup_time: length to run pipeline before beginning benchmark
    :param num_streams: number of parallel streams during async scenario
    :param thread_pinning: enable binding threads to cores
    :param engine: inference engine, deepsparse or onnxruntime
    :param quiet: lower logging verbosity
    :return: dictionary of section times for each forward pass and the total run time
    """

    if quiet:
        set_logging_level(logging.WARN)

    if num_cores is None:
        num_cores = cpu_architecture().num_available_physical_cores

    if config is None:
        _LOGGER.warning("No input configuration provided, falling back to default.")
        config = PipelineBenchmarkConfig()

    decide_thread_pinning(thread_pinning)
    scenario = parse_scenario(scenario.lower())
    scheduler = parse_scheduler(scenario)
    num_streams = parse_num_streams(num_streams, num_cores, scenario)

    input_type = config.data_type
    kwargs = config.pipeline_kwargs
    kwargs["benchmark"] = True
    pipeline = Pipeline.create(
        task=task,
        model_path=model_path,
        engine_type=engine,
        scheduler=scheduler,
        num_cores=num_cores,
        num_streams=num_streams,
        **kwargs,
    )
    inputs = create_input_schema(pipeline, input_type, batch_size, config)

    if scenario == "singlestream":
        singlestream_benchmark(pipeline, inputs, warmup_time)
        pipeline.timer_manager.clear()
        start_time = time.perf_counter()
        singlestream_benchmark(pipeline, inputs, seconds_to_run)
    elif scenario == "multistream":
        multistream_benchmark(pipeline, inputs, warmup_time, num_streams)
        pipeline.timer_manager.clear()
        start_time = time.perf_counter()
        multistream_benchmark(pipeline, inputs, seconds_to_run, num_streams)
    elif scenario == "elastic":
        multistream_benchmark(pipeline, inputs, warmup_time, num_streams)
        pipeline.timer_manager.clear()
        start_time = time.perf_counter()
        multistream_benchmark(pipeline, inputs, seconds_to_run, num_streams)
    else:
        raise Exception(f"Unknown scenario '{scenario}'")

    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    batch_times = pipeline.timer_manager.all_times
    if len(batch_times) == 0:
        raise Exception(
            "Generated no batch timings, try extending benchmark time with '--time'"
        )

    return batch_times, total_run_time, num_streams


def calculate_statistics(
    batch_times_ms: List[float], total_run_time_ms: float, num_streams: int
) -> Dict:
    percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    buckets = numpy.percentile(batch_times_ms, percentiles).tolist()
    percentiles_dict = {
        "{:2.1f}%".format(key): value for key, value in zip(percentiles, buckets)
    }

    scaled_runtime = total_run_time_ms * num_streams
    benchmark_dict = {
        "total_percentage": sum(batch_times_ms) / scaled_runtime * 100,
        "median": numpy.median(batch_times_ms),
        "mean": numpy.mean(batch_times_ms),
        "std": numpy.std(batch_times_ms),
        **percentiles_dict,
    }
    return benchmark_dict


def calculate_section_stats(
    batch_times: Dict[str, List[float]], total_run_time: float, num_streams: int
) -> Dict[str, Dict]:
    total_run_time_ms = total_run_time * 1000

    sections = {}
    for section_name in batch_times:
        times = [t * 1000 for t in batch_times[section_name]]
        sections[section_name] = calculate_statistics(
            times, total_run_time_ms, num_streams
        )

    return sections


@click.command()
@click.argument("task_name", type=str)
@click.argument("model_path", type=str)
@click.option(
    "-c",
    "--input_config",
    type=str,
    default=None,
    help="JSON file containing schema for input data",
)
@click.option(
    "-b",
    "--batch_size",
    type=int,
    default=1,
    help="The batch size to run the analysis for. Must be greater than 0",
)
@click.option(
    "-ncores",
    "--num_cores",
    type=int,
    default=cpu_architecture().num_available_physical_cores,
    help=(
        "The number of physical cores to run the analysis on, "
        "defaults to all physical cores available on the system"
    ),
)
@click.option(
    "-s",
    "--scenario",
    type=str,
    default="sync",
    help=(
        "Choose between using the async, sync and elastic scenarios. Sync and "
        "async are similar to the single-stream/multi-stream scenarios. Elastic "
        "is a newer scenario that behaves similarly to the async scenario "
        "but uses a different scheduling backend. Default value is sync."
    ),
)
@click.option(
    "-t",
    "--run_time",
    type=int,
    default=10,
    help="The number of seconds the benchmark will run. Default is 10 seconds.",
)
@click.option(
    "-w",
    "--warmup_time",
    type=int,
    default=2,
    help=(
        "The number of seconds the benchmark will warmup before running."
        "Default is 2 seconds."
    ),
)
@click.option(
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
@click.option(
    "-pin",
    "--thread_pinning",
    type=str,
    default="core",
    help=(
        "Enable binding threads to cores ('core' the default), "
        "threads to cores on sockets ('numa'), or disable ('none')"
    ),
)
@click.option(
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
@click.option(
    "-q",
    "--quiet",
    help="Lower logging verbosity",
    default=False,
)
@click.option(
    "-x",
    "--export_path",
    help="Store results into a JSON file",
    type=str,
    default=None,
)
def main(
    task_name: str,
    model_path: str,
    input_config: str,
    batch_size: int,
    num_cores: int,
    scenario: str,
    run_time: int,
    warmup_time: int,
    num_streams: int,
    thread_pinning: str,
    engine: str,
    quiet: bool,
    export_path: str,
):
    config = parse_input_config(input_config)

    _LOGGER.info("Original Model Path: %s" % model_path)
    _LOGGER.info("Task: %s" % task_name)
    _LOGGER.info("Batch Size: %d" % batch_size)
    _LOGGER.info("Scenario: %s" % scenario)
    _LOGGER.info("Requested Run Time(sec): %d" % run_time)

    batch_times, total_run_time, num_streams = benchmark_pipeline(
        model_path=model_path,
        task=task_name,
        config=config,
        batch_size=batch_size,
        num_cores=num_cores,
        scenario=scenario,
        seconds_to_run=run_time,
        warmup_time=warmup_time,
        num_streams=num_streams,
        thread_pinning=thread_pinning,
        engine=engine,
        quiet=quiet,
    )

    section_stats = calculate_section_stats(batch_times, total_run_time, num_streams)
    items_per_sec = (len(batch_times["total_inference"]) * batch_size) / total_run_time

    benchmark_results = {
        "items_per_sec": items_per_sec,
        "seconds_ran": total_run_time,
        "iterations": len(batch_times),
        "compute_sections": section_stats,
    }

    export_dict = {
        "engine": engine,
        "version": __version__,
        "model_path": model_path,
        "batch_size": batch_size,
        "num_cores": num_cores,
        "scenario": scenario,
        "seconds_to_run": run_time,
        "num_streams": num_streams,
        "input_config": dict(config),
        "benchmark_results": benchmark_results,
    }

    # Export results
    export_path = export_path
    if export_path:
        _LOGGER.info("Saving benchmark results to JSON file at %s" % export_path)
        with open(export_path, "w") as out:
            json.dump(export_dict, out, indent=2)

    # Results summary
    print("Original Model Path: %s" % model_path)
    print("Batch Size: %d" % batch_size)
    print("Scenario: %s" % scenario)
    print("Iterations: %d" % int(benchmark_results["iterations"]))
    print("Total Runtime: %.4f" % total_run_time)
    print("Throughput (items/sec): %.4f" % benchmark_results["items_per_sec"])

    print("Processing Time Breakdown: ")
    for section in section_stats:
        print("     %s: %.2f%%" % (section, section_stats[section]["total_percentage"]))

    print("Mean Latency Breakdown (ms/batch): ")
    for section in section_stats:
        print("     %s: %.4f" % (section, section_stats[section]["mean"]))


if __name__ == "__main__":
    main()
