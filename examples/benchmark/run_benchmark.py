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
Example script for benchmarking an ONNX model over random inputs and using
both the DeepSparse Engine and ONNXRuntime, comparing results.

In this method, we can assume that ONNXRuntime will give the
"correct" output as it is the industry-standard solution.

##########
Command help:
usage: run_benchmark.py [-h] [-s BATCH_SIZE] [-b NUM_ITERATIONS]
    [-w NUM_WARMUP_ITERATIONS] onnx_filepath

Benchmark an ONNX model, comparing between DeepSparse and ONNXRuntime

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file being benchmarked

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -b NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
                        The number of times the benchmark will be run
  -w NUM_WARMUP_ITERATIONS, --num_warmup_iterations NUM_WARMUP_ITERATIONS
                        The number of warmup runs that will be executed before
                        the actual benchmarking

##########
Example command for benchmarking a downloaded resnet50 model
for batch size 8, over 100 iterations:
python examples/benchmark/run_benchmark.py \
    ~/Downloads/resnet50.onnx \
    --batch_size 8 \
    --num_iterations 100
"""

import argparse
import time

import onnxruntime

from deepsparse import compile_model, cpu
from deepsparse.benchmark import BenchmarkResults
from deepsparse.utils import (
    generate_random_inputs,
    get_input_names,
    get_output_names,
    override_onnx_batch_size,
    verify_outputs,
)


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark an ONNX model, comparing between DeepSparse and ONNXRuntime"
        )
    )

    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file being benchmarked",
    )

    parser.add_argument(
        "-s",
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-b",
        "--num_iterations",
        help="The number of times the benchmark will be run",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-w",
        "--num_warmup_iterations",
        help=(
            "The number of warmup runs that will be executed before the actual"
            " benchmarking"
        ),
        type=int,
        default=5,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_iterations = args.num_iterations
    num_warmup_iterations = args.num_warmup_iterations

    inputs = generate_random_inputs(onnx_filepath, batch_size)
    input_names = get_input_names(onnx_filepath)
    output_names = get_output_names(onnx_filepath)
    inputs_dict = {name: value for name, value in zip(input_names, inputs)}

    # Benchmark ONNXRuntime
    print("Benchmarking model with ONNXRuntime...")
    sess_options = onnxruntime.SessionOptions()
    with override_onnx_batch_size(onnx_filepath, batch_size) as override_onnx_filepath:
        ort_network = onnxruntime.InferenceSession(override_onnx_filepath, sess_options)

        ort_results = BenchmarkResults()
        for i in range(num_warmup_iterations):
            ort_network.run(output_names, inputs_dict)
        for i in range(num_iterations):
            start = time.time()
            output = ort_network.run(output_names, inputs_dict)
            end = time.time()
            ort_results.append_batch(
                time_start=start, time_end=end, batch_size=batch_size, outputs=output
            )

    # Benchmark DeepSparse Engine
    print("Benchmarking model with DeepSparse Engine...")
    dse_network = compile_model(onnx_filepath, batch_size=batch_size)
    print(f"Engine info: {dse_network}")
    dse_results = dse_network.benchmark(
        inputs, num_iterations, num_warmup_iterations, include_outputs=True
    )

    for dse_output, ort_output in zip(dse_results.outputs, ort_results.outputs):
        verify_outputs(dse_output, ort_output)

    print("ONNXRuntime", ort_results)
    print()
    print("DeepSparse Engine", dse_results)


if __name__ == "__main__":
    main()
