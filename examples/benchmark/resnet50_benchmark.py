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
Example script for benchmarking sparsified ResNet50 models from
the SparseZoo on the DeepSparse Engine.

##########
Command help:
usage: resnet50_benchmark.py [-h] [-s BATCH_SIZE] [-j NUM_CORES] \
    [-b NUM_ITERATIONS] [-w NUM_WARMUP_ITERATIONS]

Benchmark sparsified ResNet50 models from the SparseZoo

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -j NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on, defaults
                        to all physical cores available on the system
  -b NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
                        The number of times the benchmark will be run
  -w NUM_WARMUP_ITERATIONS, --num_warmup_iterations NUM_WARMUP_ITERATIONS
                        The number of warmup runs that will be executed before the
                        actual benchmarking

##########
Example command for ResNet50 benchmarks with batch size 128 and 16 cores used:
python resnet50_benchmark.py \
    --batch_size 128 \
    --num_cores 16
"""

import argparse

import numpy

from deepsparse import benchmark_model, cpu


CORES_PER_SOCKET, AVX_TYPE, VNNI = cpu.cpu_details()


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Benchmark sparsified ResNet50 models from the SparseZoo")
    )

    parser.add_argument(
        "-s",
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-j",
        "--num_cores",
        type=int,
        default=CORES_PER_SOCKET,
        help=(
            "The number of physical cores to run the analysis on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-b",
        "--num_iterations",
        help="The number of times the benchmark will be run",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-w",
        "--num_warmup_iterations",
        help=(
            "The number of warmup runs that will be executed before the actual"
            " benchmarking"
        ),
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size
    num_cores = args.num_cores
    num_iterations = args.num_iterations
    num_warmup_iterations = args.num_warmup_iterations

    sample_inputs = [numpy.random.randn(batch_size, 3, 224, 224).astype(numpy.float32)]

    print(
        f"Starting DeepSparse benchmarks using batch size {batch_size} and {num_cores}"
        " cores"
    )

    results = benchmark_model(
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",
        sample_inputs,
        batch_size=batch_size,
        num_cores=num_cores,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    print(f"ResNet-50 v1 Dense FP32 {results}")

    results = benchmark_model(
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned-conservative"
        ),
        sample_inputs,
        batch_size=batch_size,
        num_cores=num_cores,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    print(f"ResNet-50 v1 Pruned Conservative FP32 {results}")

    results = benchmark_model(
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned-moderate"
        ),
        sample_inputs,
        batch_size=batch_size,
        num_cores=num_cores,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    print(f"ResNet-50 v1 Pruned Moderate FP32 {results}")

    if not VNNI:
        print(
            "WARNING: VNNI instructions not detected, "
            "quantization (INT8) speedup not well supported"
        )

    results = benchmark_model(
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned_quant-moderate"
        ),
        sample_inputs,
        batch_size=batch_size,
        num_cores=num_cores,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    print(f"ResNet-50 v1 Pruned Moderate INT8 {results}")

    results = benchmark_model(
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet-augmented/"
            "pruned_quant-aggressive"
        ),
        sample_inputs,
        batch_size=batch_size,
        num_cores=num_cores,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
    )
    print(f"ResNet-50 v1 Pruned Aggressive INT8 {results}")


if __name__ == "__main__":
    main()
