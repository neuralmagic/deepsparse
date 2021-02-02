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
Example script for downloading a classification model from SparseZoo with real data
and using the DeepSparse Engine for inference and benchmarking.

##########
Command help:
usage: classification.py [-h] [-s BATCH_SIZE] [-j NUM_CORES]
    {efficientnet_b0,efficientnet_b4,inception_v3,mobilenet_v1,mobilenet_v2,resnet_101,
    resnet_101_2x,resnet_152,resnet_18,resnet_34,resnet_50,resnet_50_2x,vgg_11,vgg_11bn,
    vgg_13,vgg_13bn,vgg_16,vgg_16bn,vgg_19,vgg_19bn}

Benchmark and check accuracy of image classification models

positional arguments:
  {efficientnet_b0,efficientnet_b4,inception_v3,mobilenet_v1,mobilenet_v2,resnet_101,
  resnet_101_2x,resnet_152,resnet_18,resnet_34,resnet_50,resnet_50_2x,vgg_11,vgg_11bn,
  vgg_13,vgg_13bn,vgg_16,vgg_16bn,vgg_19,vgg_19bn}
                        Model type to analyze

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -j NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system

##########
Example command for running a mobilenet_v2 model with batch size 8 and 4 cores used:
python examples/classification/classification.py \
    mobilenet_v2 \
    --batch_size 8 \
    --num_cores 4
"""

import argparse
from inspect import getmembers, isfunction

import numpy

from deepsparse import compile_model, cpu
from deepsparse.utils import verify_outputs
from sparsezoo.models import classification
from sparsezoo.objects import Model


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()

model_registry = dict(getmembers(classification, isfunction))


def fetch_model(model_name: str) -> Model:
    if model_name not in model_registry:
        raise Exception(
            f"Could not find model '{model_name}' in classification model registry."
        )
    return model_registry[model_name]()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and check accuracy of image classification models"
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


def calculate_top1_accuracy(pred: numpy.array, labels: numpy.array) -> float:
    """
    :param pred: the model's prediction to compare with
    :param labels: the labels for the data to compare to
    :return: the calculated top1 accuracy
    """
    batch_size = pred.shape[0]
    pred = numpy.argmax(pred, axis=-1)
    labels = numpy.argmax(labels, axis=-1)

    correct = (pred == labels).sum()
    correct *= 100.0 / batch_size

    return correct


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

    if "labels" in batch:
        batched_labels = batch["labels"]
        # Measure accuracy against ground truth labels
        accuracy = calculate_top1_accuracy(predicted_outputs[-1], batched_labels[0])
        print("Top-1 Accuracy for batch size {}: {:.2f}%".format(batch_size, accuracy))

    # BENCHMARK
    # Record output from executing through the DeepSparse engine
    print("Benchmarking...")
    results = engine.benchmark(batched_inputs)
    print(results)


if __name__ == "__main__":
    main()
