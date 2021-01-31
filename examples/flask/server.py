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
Example script for hosting an ONNX model as a Flask server
using the DeepSparse Engine as the inference backend

##########
Command help:
usage: server.py [-h] [-s BATCH_SIZE] [-j NUM_CORES] [-a ADDRESS] [-p PORT]
 onnx_filepath

Host an ONNX model as a server, using the DeepSparse Engine and Flask

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -j NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system
  -a ADDRESS, --address ADDRESS
                        The IP address of the hosted model
  -p PORT, --port PORT  The port that the model is hosted on

##########
Example command for hosting a downloaded ResNet-50 model:
python examples/flask/server.py \
    ~/Downloads/resnet50.onnx
"""

import argparse

import flask
from flask_cors import CORS

from deepsparse import compile_model
from utils_flask import bytes_to_tensors, tensors_to_bytes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Host an ONNX model as a server, using the DeepSparse Engine and Flask"
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
        "-j",
        "--num_cores",
        type=int,
        default=0,
        help=(
            "The number of physical cores to run the analysis on, "
            "defaults to all physical cores available on the system"
        ),
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


def create_model_inference_app(
    model_path: str, batch_size: int, num_cores: int, address: str, port: str
) -> flask.Flask:
    print(f"Compiling model at {model_path}")
    engine = compile_model(model_path, batch_size, num_cores)
    print(engine)

    app = flask.Flask(__name__)
    CORS(app)

    @app.route("/predict", methods=["POST"])
    def predict():
        data = flask.request.get_data()

        inputs = bytes_to_tensors(data)
        print(f"Received {len(inputs)} inputs from client")

        print("Executing model")
        outputs, elapsed_time = engine.timed_run(inputs)

        print(f"Inference time took {elapsed_time * 1000.0:.4f} milliseconds")
        print(f"Produced {len(outputs)} output tensors")
        return tensors_to_bytes(outputs)

    @app.route("/info", methods=["GET"])
    def info():
        return flask.jsonify({"model_path": model_path, "engine": repr(engine)})

    print("Starting Flask app")
    app.run(host=address, port=port, debug=False, threaded=True)


def main():
    args = parse_args()
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_cores = args.num_cores
    address = args.address
    port = args.port

    create_model_inference_app(onnx_filepath, batch_size, num_cores, address, port)


if __name__ == "__main__":
    main()
