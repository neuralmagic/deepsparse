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
usage: client.py [-h] [-s BATCH_SIZE] [-a ADDRESS] [-p PORT] onnx_filepath

Communicate with a Flask server hosting an ONNX model with the
DeepSparse Engine as inference backend.

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
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
import time

import requests

from deepsparse.utils import generate_random_inputs
from utils_flask import bytes_to_tensors, tensors_to_bytes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Communicate with a Flask server hosting an ONNX model with the DeepSparse"
            " Engine as inference backend."
        )
    )

    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file",
    )

    parser.add_argument(
        "-s",
        "--batch_size",
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
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    address = args.address
    port = args.port

    prediction_url = f"http://{address}:{port}/predict"

    inputs = generate_random_inputs(onnx_filepath, batch_size)

    print(f"Sending {len(inputs)} input tensors to {prediction_url}")

    start = time.time()
    # Encode inputs
    data = tensors_to_bytes(inputs)
    # Send data to server for inference
    response = requests.post(prediction_url, data=data)
    # Decode outputs
    outputs = bytes_to_tensors(response.content)
    end = time.time()
    elapsed_time = end - start

    print(f"Received response of {len(outputs)} output tensors:")
    print(f"Round-trip time took {elapsed_time * 1000.0:.4f} milliseconds")

    for i, out in enumerate(outputs):
        print(f"    output #{i}: shape {out.shape}")


if __name__ == "__main__":
    main()
