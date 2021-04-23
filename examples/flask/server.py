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

from deepsparse import Scheduler, compile_model
from deepsparse.utils import arrays_to_bytes, bytes_to_arrays


def engine_flask_server(
    model_path: str,
    batch_size: int = 1,
    num_cores: int = None,
    num_sockets: int = None,
    scheduler: Scheduler = None,
    address: str = "0.0.0.0",
    port: str = "5543",
) -> flask.Flask:
    """

    :param model_path: Either a path to the model's onnx file, a SparseZoo model stub
        prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
        object that defines the neural network
    :param batch_size: The batch size of the inputs to be used with the model
    :param num_cores: The number of physical cores to run the model on.
        Pass None or 0 to run on the max number of cores
        in one socket for the current machine, default None
    :param num_sockets: The number of physical sockets to run the model on.
        Pass None or 0 to run on the max number of sockets for the
        current machine, default None
    :param scheduler: The kind of scheduler to execute with. Pass None for the default.
    :param address: IP address to run on. Default is 0.0.0.0
    :param port: port to run on. Default is 5543
    :return: launches a flask server on the given address and port can run the
        given model on the DeepSparse engine via HTTP requests
    """
    print(f"Compiling model at {model_path}")
    engine = compile_model(model_path, batch_size, num_cores, num_sockets, scheduler)
    print(engine)

    app = flask.Flask(__name__)
    CORS(app)

    @app.route("/run", methods=["POST"])
    def run():
        data = flask.request.get_data()

        inputs = bytes_to_arrays(data)
        print(f"Received {len(inputs)} inputs from client")

        print("Executing model")
        outputs, elapsed_time = engine.timed_run(inputs)

        print(f"Inference time took {elapsed_time * 1000.0:.4f} milliseconds")
        print(f"Produced {len(outputs)} output tensors")
        return arrays_to_bytes(outputs)

    @app.route("/info", methods=["GET"])
    def info():
        return flask.jsonify({"model_path": model_path, "engine": repr(engine)})

    print("Starting Flask app")
    app.run(host=address, port=port, debug=False, threaded=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Host an ONNX model as a server, using the DeepSparse Engine and Flask"
        )
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub for the model",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to run the engine with",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=0,
        help=(
            "The number of physical cores to run the engine on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-s",
        "--num-sockets",
        type=int,
        default=None,
        help=(
            "The number of physical sockets to run the engine on, "
            "defaults to all physical sockets available on the system"
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="The kind of scheduler to run with",
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

    engine_flask_server(
        args.model_path,
        args.batch_size,
        args.num_cores,
        args.num_sockets,
        args.scheduler,
        args.address,
        args.port,
    )


if __name__ == "__main__":
    main()
