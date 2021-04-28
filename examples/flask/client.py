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
Example script for communicating with a Flask server hosting an
ONNX model with the DeepSparse Engine as inference backend.

##########
Command help:
usage: client.py [-h] [-b BATCH_SIZE] [-a ADDRESS] [-p PORT] model_path

Communicate with a Flask server hosting an ONNX model with the DeepSparse
Engine as inference backend.

positional arguments:
  model_path            The full filepath of the ONNX model file or SparseZoo
                        stub of model

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to run the analysis for
  -a ADDRESS, --address ADDRESS
                        The IP address of the hosted model
  -p PORT, --port PORT  The port that the model is hosted on

##########
Example command for communicating with a ResNet-50 model already hosted with server.py:
python examples/flask/client.py \
    ~/Downloads/resnet50.onnx
"""

import argparse
import os
import time
from typing import Any, Callable, List

import numpy
import requests

from deepsparse.utils import (
    arrays_to_bytes,
    bytes_to_arrays,
    generate_random_inputs,
    log_init,
)


_LOGGER = log_init(os.path.basename(__file__))


class EngineFlaskClient:
    """
    Client object for interacting with HTTP server invoked with `engine_flask_server`.

    :param address: IP address of server to query
    :param port: port that the server is running on
    :param preprocessing_fn: function to preprocess inputs to the run argument before
        sending inputs to the model server. Defaults to the `arrays_to_bytes` function
        for serializing lists of numpy arrays
    :param preprocessing_fn: function to postprocess outputs from model server
        inferences. Defaults to the `bytes_to_arrays` function for de-serializing
        lists of numpy arrays
    """

    def __init__(
        self,
        address: str,
        port: str,
        preprocessing_fn: Callable[[Any], Any] = arrays_to_bytes,
        postprocessing_fn: Callable[[Any], Any] = bytes_to_arrays,
    ):
        self.url = f"http://{address}:{port}"
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn

    def run(self, inp: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        Client function for running a forward pass of the server model.

        :param inp: the list of inputs to pass to the server for inference.
            The expected order is the inputs order as defined in the ONNX graph
        :return: the list of outputs from the server after executing over the inputs
        """
        data = self.preprocessing_fn(inp)
        response = self._post("run", data=data)
        return self.postprocessing_fn(response)

    def _post(self, route: str, data: Any):
        route_url = f"{self.url}/{route}"
        _LOGGER.debug(f"Sending POST request to {route_url}")
        return requests.post(route_url, data=data).content


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Communicate with a Flask server hosting an ONNX model with the DeepSparse"
            " Engine as inference backend."
        )
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub of model",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        default="0.0.0.0",
        help="The IP address of the hosted model",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="5543",
        help="The port that the model is hosted on",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    engine = EngineFlaskClient(args.address, args.port)

    inputs = generate_random_inputs(args.model_path, args.batch_size)

    _LOGGER.info(f"Sending {len(inputs)} input tensors to {engine.url}/run")

    start = time.time()
    outputs = engine.run(inputs)
    end = time.time()
    elapsed_time = end - start

    _LOGGER.info(f"Round-trip time took {elapsed_time * 1000.0:.4f} milliseconds")
    _LOGGER.info(f"Received response of {len(outputs)} output tensors:")

    for i, out in enumerate(outputs):
        _LOGGER.info(f"\toutput #{i}: shape {out.shape}")


if __name__ == "__main__":
    main()
