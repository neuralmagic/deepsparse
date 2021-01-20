"""
Example script for hosting an ONNX model as a Flask server
using the DeepSparse Engine as the inference backend

##########
Command help:
usage: server.py [-h] [-s BATCH_SIZE] [-j NUM_CORES] onnx_filepath

Host an ONNX model as a server, using the DeepSparse Engine and Flask

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file being benchmarked

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -j NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on, defaults to all physical cores
                        available on the system

##########
Example command for hosting a downloaded resnet50 model for batch size 8 and 4 cores:
python examples/flask/server.py \
    ~/Downloads/resnet50.onnx \
    --batch_size 8 \
    --num_cores 4
"""

import argparse

import numpy

from deepsparse import compile_model
from flask import Flask, jsonify, request
from flask_cors import CORS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Host an ONNX model as a server, using the DeepSparse Engine and Flask"
    )

    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file being benchmarked",
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
        help="The number of physical cores to run the analysis on, "
        "defaults to all physical cores available on the system",
    )

    return parser.parse_args()


def create_model_inference_app(
    model_path: str, batch_size: int, num_cores: int
) -> Flask:
    print(f"compiling model at {model_path}")
    engine = compile_model(model_path, batch_size, num_cores)
    app = Flask(__name__)
    CORS(app)

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        inputs = []
        for inp in data["inputs"]:
            inputs.append(numpy.array(inp).astype(numpy.float32))

        outputs = engine.run(inputs)
        outputs = [out.tolist() for out in outputs]

        return jsonify({"outputs": outputs})

    @app.route("/info", methods=["GET"])
    def info():
        return jsonify({"model_path": model_path, "engine": repr(engine)})

    print(f"starting app")
    app.run(host="0.0.0.0", port="5543", debug=True, threaded=True)


def main():
    args = parse_args()
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_cores = args.num_cores

    create_model_inference_app(onnx_filepath, batch_size, num_cores)


if __name__ == "__main__":
    main()
