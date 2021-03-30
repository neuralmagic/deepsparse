'''
Example created by Adriano A. Santos (https://github.com/adrianosantospb).
'''
import flask
from flask import request
from flask_cors import CORS
import numpy as np
import cv2
from utils.util import preprocessing

from deepsparse import compile_model, cpu
from deepsparse.utils import arrays_to_bytes, bytes_to_arrays


def create_model_inference_app(
    model_path: str, batch_size: int, num_cores: int, address: str, port: str, image_size:int
) -> flask.Flask:
    print(f"Compiling model at {model_path}")
    engine = compile_model(model_path, batch_size, num_cores)
    print(engine)

    app = flask.Flask(__name__)
    CORS(app)

    @app.route("/api/predict", methods=["POST"])
    def predict():
        req = request
        nparr = np.fromstring(req.data, np.uint8)
        img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Image preparation
        img = preprocessing(img0, image_size)
        # Predictions
        outputs = engine.run([img])
        #print(f"Inference time took {elapsed_time * 1000.0:.4f} milliseconds")
        return arrays_to_bytes(outputs)

    @app.route("/api/info", methods=["GET"])
    def info():
        return flask.jsonify({"model_path": model_path, "engine": repr(engine)})

    print("Starting Flask app")
    app.run(host=address, port=port, debug=False, threaded=True)


def main():
    # =====================================================
    # Define the number of cores to use
    # =====================================================
    num_cores, _, _ = cpu.cpu_details()
    batch_size = 1
    # Model file path
    onnx_filepath = "model/model.onnx"
    # Server name
    address = "localhost"
    # Port
    port = "9898"
    # Image size
    image_size = 416
    create_model_inference_app(onnx_filepath, batch_size, num_cores, address, port, image_size)

if __name__ == "__main__":
    main()
