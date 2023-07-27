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
import glob

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

def get_input_schema_type(pipeline: Pipeline) -> str:
    input_schema_requirements = list(pipeline.input_schema.__annotations__.keys())
    image_requirements = ["images"]
    basic_text_requirements = ["sequences"]
    question_requirements = ["question", "context", "id"]
    text_generation_requirements = ["sequences", "return_logits", "session_id", "fixed_sequences_length"]

    if input_schema_requirements == image_requirements or "YOLO" in pipeline.input_schema.__name__:
        return "image"
    elif input_schema_requirements == basic_text_requirements:
        return "text"
    elif input_schema_requirements == question_requirements:
        return "question"
    elif input_schema_requirements == text_generation_requirements:
        return "text_generation"

    raise Exception("Unknown schema requirement {}".format(input_schema_requirements))

def generate_image_data(config: Dict, batch_size: int) -> List[numpy.ndarray]:
    input_data = []
    if "input_image_shape" in config and len(config["input_image_shape"]) == 3:
        image_shape = config["input_image_shape"]
    else:
        image_shape = (240, 240, 3)
        _LOGGER.warning("Using default image shape {}".format(image_shape))

    for _ in range(batch_size):
        rand_array = numpy.random.randint(0,high=255, size=image_shape).astype(numpy.uint8)
        input_data.append(rand_array)

    return input_data

def load_image_data(config: Dict, batch_size: int) -> List[str]:
    path_to_data = config["data_folder"]
    recursive_search = config["recursive_search"]
    files = []
    for f in glob.glob(path_to_data + "/**", recursive=recursive_search):
        if f.lower().endswith(".jpeg"):
            files.append(f)
    if len(files) < batch_size:
        raise Exception("Not enough images found in {}".format(path_to_data))
    input_data = random.sample(files, batch_size)

    return input_data

def generate_text_data(config: Dict, batch_size: int) -> List[str]:
    input_data = []
    if 'gen_sequence_length' in config:
        string_length = config['gen_sequence_length']
    else:
        string_length = 100
        _LOGGER.warning("Using default string length {}".format(string_length))
    for _ in range(batch_size):
        rand_sentence = generate_sentence(string_length)
        input_data.append(rand_sentence)
    
    return input_data

def load_text_data(config: Dict, batch_size: int) -> List[str]:
    path_to_data = config["data_folder"]
    recursive_search = config["recursive_search"]
    files = []
    for f in glob.glob(path_to_data + "/**", recursive=recursive_search):
        if f.lower().endswith(".txt"):
            files.append(f)
    if len(files) < batch_size:
        raise Exception("Not enough images found in {}".format(path_to_data))
    input_files = random.sample(files, batch_size)
    if "max_string_length" in config:
        max_string_length = config["max_string_length"]
    else:
        max_string_length = -1
        _LOGGER.warning("Using default max string length {}".format(max_string_length))
    input_data = []
    for f_path in input_files:
        f = open(f_path)
        text_data = f.read()
        f.close()
        input_data.append(text_data[:max_string_length])
    return input_data

def generate_sentence(string_length: int, avg_word_length: int = 5):
    random_chars = ''.join(random.choices(string.ascii_letters, k=string_length))
    space_locations = random.sample(range(string_length), int(string_length / avg_word_length))
    random_chars = list(random_chars)
    for loc in space_locations:
        random_chars[loc] = ' '
    return ''.join(random_chars)

def generate_question_data(config: Dict) -> Tuple[str, str]:
    if 'gen_sequence_length' in config:
        string_length = config['gen_sequence_length']
    else:
        string_length = 100
        _LOGGER.warning("Using default string length {}".format(string_length))
    question = generate_sentence(string_length)
    context = generate_sentence(string_length)
    return (question, context)

def load_question_data(config: Dict) -> Tuple[str, str]:
    path_to_questions = config["question_file"]
    path_to_context = config["context_file"]

    f_question = open(path_to_questions)
    f_context = open(path_to_context)
    question = f_question.read()
    context = f_context.read()
    f_question.close()
    f_context.close()
    return question, context

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
    input_schema_requirement = get_input_schema_type(pipeline)

    if input_type == "dummy":
        if input_schema_requirement == "image":
            input_data = generate_image_data(config, batch_size)
            inputs = pipeline.input_schema(images=input_data)
        elif input_schema_requirement == "text":
            input_data = generate_text_data(config, batch_size)
            inputs = pipeline.input_schema(sequences=input_data)
        elif input_schema_requirement == "question":
            _LOGGER.warn("Only batch size of 1 supported for Question Answering Pipeline")
            question, context = generate_question_data(config)
            inputs = pipeline.input_schema(question=question, context=context)
        elif input_schema_requirement == "text_generation":
            seqs = generate_text_data(config, batch_size)
            fix_len = config["fix_sequence_length"]
            inputs = pipeline.input_schema(sequences=seqs, return_logits=False, session_id=None, fixed_sequences_length=fix_len)
    elif input_type == "real":
        if input_schema_requirement == "image":
            input_data = load_image_data(config, batch_size)
            inputs = pipeline.input_schema(images=input_data)
        elif input_schema_requirement == "text":
            input_data = load_text_data(config, batch_size)
            inputs = pipeline.input_schema(sequences=input_data)
        elif input_schema_requirement == "question":
            _LOGGER.warn("Only batch size of 1 supported for Question Answering Pipeline")
            question, context = load_question_data(config)
            inputs = pipeline.input_schema(question=question, context=context)
        elif input_schema_requirement == "text_generation":
            seqs = load_text_data(config, batch_size)
            fix_len = config["fix_sequence_length"]
            inputs = pipeline.input_schema(sequences=seqs, return_logits=False, session_id=None, fixed_sequences_length=fix_len)
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

    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    if len(batch_times) == 0:
        raise Exception("Generated no batch timings, try extending benchmark time with '--time'")

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

def calculate_section_stats(batch_times: List[StagedTimer], total_run_time: float) -> Dict[str, Dict]:
    compute_sections = batch_times[0].stages
    total_run_time_ms = total_run_time * 1000

    sections = {}
    for section in compute_sections:
        section_times = [st.times[section] * 1000 for st in batch_times]
        sections[section] = calculate_statistics(section_times, total_run_time_ms)

    return sections


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

    section_stats = calculate_section_stats(batch_times, total_run_time)
    items_per_sec = (len(batch_times) * args.batch_size) / total_run_time


    export_dict = {
        "scenario": args.scenario,
        "items_per_sec": items_per_sec,
        "seconds_ran": total_run_time,
        "iterations": len(batch_times),
        "compute_sections": section_stats
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
    compute_sections = batch_times[0].stages
    for section in compute_sections:
        print("     {}: {:.2f}%".format(section, section_stats[section]["total_percentage"]))
    
    print("Mean Latency Breakdown (ms/batch): ")
    for section in compute_sections:
        print("     {}: {:.4f}".format(section, section_stats[section]["mean"]))

if __name__ == "__main__":
    main()
