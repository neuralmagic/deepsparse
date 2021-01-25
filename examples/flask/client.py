"""
Example script for communicating with a Flask server hosting an
ONNX model with the DeepSparse Engine as inference backend.

##########
Command help:
usage: client.py [-h] [-s BATCH_SIZE] onnx_filepath

Communicate with a Flask server hosting an ONNX model with the DeepSparse Engine as inference backend.

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for

##########
Example command for communicating with a hosted resnet50 model:
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
        description="Communicate with a Flask server hosting an ONNX model with the DeepSparse Engine as inference backend."
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

    return parser.parse_args()


def main():
    args = parse_args()
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size

    inputs = generate_random_inputs(onnx_filepath, batch_size)

    server_url = "http://0.0.0.0:5543/predict"
    print(f"Sending {len(inputs)} input tensors to {server_url}")

    start = time.time()
    # Encode inputs
    data = tensors_to_bytes(inputs)
    # Send data to server for inference
    response = requests.post(server_url, data=data)
    # Decode outputs
    outputs = bytes_to_tensors(response.content)
    end = time.time()
    elapsed_time = end - start

    print(f"Recieved response of {len(outputs)} output tensors:")
    print(f"Round-trip time took {elapsed_time * 1000.0:.4f} milliseconds")

    for i, out in enumerate(outputs):
        print(f"    output #{i}: shape {out.shape}")


if __name__ == "__main__":
    main()
