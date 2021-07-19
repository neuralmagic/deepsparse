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
Example script for hosting a BERT ONNX model as a Flask server
using the DeepSparse Engine as the inference backend

##########
Command help:
usage: server.py [-h] [-b BATCH_SIZE] [-c NUM_CORES] [-a ADDRESS] [-p PORT]
                 onnx_filepath

Host a Torchvision ONNX classification model as a server, using the DeepSparse
Engine and Flask

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file or SparseZoo
                        stub to the model

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to run the analysis for
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system
  -a ADDRESS, --address ADDRESS
                        The IP address of the hosted model
  -p PORT, --port PORT  The port that the model is hosted on

##########
Example command for running using a model from sparsezoo:
python server.py \
    zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate
"""
import argparse
import time
from typing import Any, Callable

import flask
from flask import Flask, jsonify, make_response
from flask_cors import CORS

from deepsparse import compile_model
from deepsparse.utils import arrays_to_bytes, bytes_to_arrays


app = Flask(__name__)
CORS(app)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Host a Torchvision ONNX classification model as a server, "
            "using the DeepSparse Engine and Flask"
        )
    )
    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub to the model",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to run the analysis for",
    )

    parser.add_argument(
        "-c",
        "--num-cores",
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


def run_server(
    predictor: Callable[[Any, Any], Any],
    host: str = "0.0.0.0",
    port: str = "5543",
    info: str = None,
):
    """
    Method to create routes and serve predictor on the specified port
    """
    assert callable(predictor), "predictor should be callable"

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Expects data as bytes
        :returns: returns classification result as bytes
        """
        start = time.time()
        raw_data = flask.request.get_data()
        inputs = bytes_to_arrays(raw_data)
        print(f"Received {len(inputs)} images from client")

        result = predictor.timed_run(inputs)
        inference_time = (time.time() - start) * 1000

        print(f"inference time: {inference_time:.4f} ms")
        return arrays_to_bytes(result[0])

    @app.route("/info", methods=["GET"])
    def information():
        """
        Route for model information
        """
        return jsonify(
            {
                "model_path": info,
                "engine": predictor,
            }
        )

    @app.route("/<page_name>")
    def unimplemented(page_name: str):
        """
        Route for unimplemented pages
        """
        routes = [
            rule
            for rule in app.url_map.iter_rules()
            if "static" not in rule.endpoint and "page_name" not in rule.endpoint
        ]
        return make_response(
            (
                f"The page named {page_name} has not been implemented. Supported "
                f"routes include {routes}"
            ),
            404,
        )

    print("Starting Flask app")
    app.run(host=host, port=port, debug=False, threaded=True)


def main():
    """
    process arguments and run server
    """
    _config = parse_args()
    # Get model
    print(f"Compiling model at {_config.onnx_filepath}")
    engine = compile_model(_config.onnx_filepath, _config.batch_size, _config.num_cores)

    print(engine)

    # Serve model
    run_server(
        predictor=engine,
        host=_config.address,
        port=_config.port,
        info=_config.onnx_filepath,
    )


if __name__ == "__main__":
    main()
