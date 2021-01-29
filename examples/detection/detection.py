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
Example script for downloading an object detection model from SparseZoo with real data
and using the DeepSparse Engine for inference and benchmarking.

##########
Command help:
usage: detection.py [-h] [-s BATCH_SIZE] [-j NUM_CORES] {ssd_resnet50_300,yolo_v3}

Benchmark and check accuracy of object detection models

positional arguments:
  {ssd_resnet50_300,yolo_v3}
                        Model type to analyze

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -j NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system

##########
Example command for running a yolo_v3 model with batch size 8 and 4 cores used:
python examples/detection/detection.py \
    yolo_v3 \
    --batch_size 8 \
    --num_cores 4
"""

import argparse
from inspect import getmembers, isfunction

from deepsparse import compile_model, cpu
from deepsparse.utils import verify_outputs
from sparsezoo.models import detection
from sparsezoo.objects import Model


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()

model_registry = dict(getmembers(detection, isfunction))


def fetch_model(model_name: str) -> Model:
    if model_name not in model_registry:
        raise Exception(
            f"Could not find model '{model_name}' in detection model registry."
        )
    return model_registry[model_name]()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and check accuracy of object detection models"
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=model_registry.keys(),
        help="Model type to analyze",
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

    return parser.parse_args()


def main():
    args = parse_args()
    model = fetch_model(args.model_name)
    batch_size = args.batch_size
    num_cores = args.num_cores

    # Gather batch of data
    batch = model.sample_batch(batch_size=batch_size)
    batched_inputs = batch["inputs"]
    batched_outputs = batch["outputs"]

    # Compile model for inference
    print("Compiling {} model with DeepSparse Engine".format(model.architecture_id))
    engine = compile_model(model, batch_size, num_cores)
    print(engine)

    # INFERENCE
    # Record output from inference through the DeepSparse Engine
    print("Executing...")
    predicted_outputs = engine(batched_inputs)

    # Compare against reference model output
    verify_outputs(predicted_outputs, batched_outputs)

    # BENCHMARK
    # Record output from executing through the DeepSparse engine
    print("Benchmarking...")
    results = engine.benchmark(batched_inputs)
    print(results)


if __name__ == "__main__":
    main()
