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
Example script for hosting a YOLO ONNX model as a Flask server
using the DeepSparse Engine as the inference backend

##########
Command help:
usage: server.py [-h] [-b BATCH_SIZE] [-c NUM_CORES] [-a ADDRESS] [-p PORT]
                 [-q] [--model-config MODEL_CONFIG]
                 onnx_filepath

Host a Yolo ONNX model as a server, using the DeepSparse Engine and Flask

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
  -q, --quantized-inputs
                        Set flag to execute inferences with int8 inputs
                        instead of float32
  --model-config MODEL_CONFIG
                        YOLO config YAML file to override default anchor
                        points when post-processing. Defaults to use standard
                        YOLOv3/YOLOv5 anchors

##########
Example command for running from a local YOLOv3 file:
python server.py \
    ~/models/yolo-v3-pruned.onnx

##########
Example command for running from a pruned YOLOv5s file:
python server.py \
    zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96
"""

import argparse
import time

import flask
import numpy
from flask_cors import CORS

from deepsparse import compile_model
from deepsparse.utils import arrays_to_bytes, bytes_to_arrays
from deepsparse_utils import (
    YoloPostprocessor,
    download_model_if_stub,
    postprocess_nms,
    yolo_onnx_has_postprocessing,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Host a Yolo ONNX model as a server, using the DeepSparse Engine and Flask"
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
    parser.add_argument(
        "-q",
        "--quantized-inputs",
        help="Set flag to execute inferences with int8 inputs instead of float32",
        action="store_true",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help=(
            "YOLO config YAML file to override default anchor points when "
            "post-processing. Defaults to use standard YOLOv3/YOLOv5 anchors"
        ),
    )

    return parser.parse_args()


def create_and_run_model_server(
    args, model_path: str, batch_size: int, num_cores: int, address: str, port: str
) -> flask.Flask:
    model_path = download_model_if_stub(model_path)
    has_postprocessing = yolo_onnx_has_postprocessing(model_path)

    print(f"Compiling model at {model_path}")
    engine = compile_model(model_path, batch_size, num_cores)
    print(engine)

    postprocessor = (
        YoloPostprocessor(cfg=args.model_config) if not has_postprocessing else None
    )

    app = flask.Flask(__name__)
    CORS(app)

    @app.route("/predict", methods=["POST"])
    def predict():
        # load raw images
        raw_data = flask.request.get_data()
        inputs = bytes_to_arrays(raw_data)
        print(f"Received {len(inputs)} images from client")

        # pre-processing
        preprocess_start_time = time.time()
        if not args.quantized_inputs:
            inputs = [inputs[0].astype(numpy.float32) / 255.0]
        preprocess_time = time.time() - preprocess_start_time
        print(f"Pre-processing time: {preprocess_time * 1000.0:.4f}ms")

        # inference
        print("Executing model")
        outputs, elapsed_time = engine.timed_run(inputs)
        print(f"Inference time: {elapsed_time * 1000.0:.4f}ms")

        # post-processing
        if postprocessor:
            postprocess_start_time = time.time()
            outputs = postprocessor.pre_nms_postprocess(outputs)
            postprocess_time = time.time() - postprocess_start_time
            print(f"Post-processing, pre-nms time: {postprocess_time * 1000.0:.4f}ms")
        else:
            outputs = outputs[0]  # post-processed values stored in first output

        # NMS
        nms_start_time = time.time()
        outputs = postprocess_nms(outputs)
        nms_time = time.time() - nms_start_time
        print(f"nms time: {nms_time * 1000.0:.4f}ms")

        return arrays_to_bytes(outputs)

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

    create_and_run_model_server(
        args, onnx_filepath, batch_size, num_cores, address, port
    )


if __name__ == "__main__":
    main()
