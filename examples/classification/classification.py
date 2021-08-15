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
import logging
from inspect import getmembers, isfunction

import numpy

from deepsparse import compile_model, cpu
from deepsparse.utils import verify_outputs
from sparsezoo.models import classification
from sparsezoo.objects import Model


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()

# Get the top-level logger object
log = logging.getLogger()

console = logging.StreamHandler()
log.addHandler(console)


class Predictor:
    model_registry = dict(getmembers(classification, isfunction))

    def __init__(self, model_name, batch_size, num_cores):
        self._model = self._fetch_model(model_name=model_name)
        self.batch_size = batch_size
        self._num_cores = num_cores
        self._engine = self._compile()

    def sample_batch(self):
        """
        Gather a batch of sample inputs, outputs and labels if available for the model
        """
        batch = self._model.sample_batch(batch_size=self.batch_size)
        return batch["inputs"], batch["outputs"], batch.get("labels")

    def predict(self, inputs):
        """
        Utility method to run inputs through the DeepSparse engine

        :param inputs: numpy array of inputs
        """
        return self._engine(inputs)

    def predict_and_verify(self, inputs, outputs):
        """
        Run the inputs through DeepSparse engine and
        Compare results against reference model output

        :param inputs: numpy array of inputs
        :param outputs: reference outputs to verify predictions
        """

        _predictions = self.predict(inputs=inputs)
        verify_outputs(_predictions, outputs)
        return _predictions

    def benchmark_on_sample_data(self):
        """
        Benchmark DeepSparse engine on sample data and return results
        """
        sample_inputs, _, _ = self.sample_batch()
        return self.benchmark(batched_inputs=sample_inputs)

    def benchmark(self, batched_inputs):
        """
        Benchmark DeepSparse Engine on a given batch of inputs

        :returns: Results from ben
        """
        return self._engine.benchmark(batched_inputs)

    def _fetch_model(self, model_name: str) -> Model:
        if model_name not in self.model_registry:
            raise Exception(
                f"Could not find model '{model_name}' in classification model registry."
            )
        return self.model_registry[model_name]()

    def _compile(self):
        # Compile model for inference
        log.info(
            f"Compiling {self._model.architecture_id} model with DeepSparse Engine"
        )
        engine = compile_model(self._model, self.batch_size, self._num_cores)
        log.info(f"Compiled Model: {engine}")
        return engine

    @staticmethod
    def top1_accuracy(pred: numpy.array, labels: numpy.array) -> float:
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


def parse_args(args=None):
    """
    Add and parse arguments

    :params args: optional arguments to parse; if None,
        arguments are read from command line
    :returns: argparse.NameSpace object containing args
    """
    parser = argparse.ArgumentParser(
        description="Benchmark and check accuracy of image classification models"
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=Predictor.model_registry.keys(),
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

    return parser.parse_args(args)


def main(args=None):
    """
    Main driver function that runs predictions,
    and benchmarks the DeepSparse engine

    :params args: optional arguments; if None,
        arguments are read from command line
    """
    args = parse_args(args)
    predictor = Predictor(args.model_name, args.batch_size, args.num_cores)
    inputs, outputs, labels = predictor.sample_batch()

    if labels:
        predictions = predictor.predict(inputs=inputs)
        # Measure accuracy against ground truth labels
        accuracy = Predictor.top1_accuracy(predictions[-1], labels[0])
        log.info(
            "Top-1 Accuracy for batch size {}: {:.2f}%".format(
                predictor.batch_size, accuracy
            )
        )

    # BENCHMARK
    # Record output from executing through the DeepSparse engine
    results = predictor.benchmark_on_sample_data()
    log.info(results)


def sanity_check():
    """
    Dummy test function
    """
    test_args = "mobilenet_v2 --batch_size 8 --num_cores 4".split()
    main(args=test_args)


if __name__ == "__main__":
    # sanity_check()
    main()
