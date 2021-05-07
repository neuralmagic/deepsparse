# Using the YOLO with DeepSparse (Neuro Magic)

This directory holds an example server and client implementation to YOLO architecture created by Adriano A. Santos (https://github.com/adrianosantospb).

Get the model.onnx file from the sparsify project and put it on the model folder (it's not mandatory). If you don't know where is it, you need to run the *sparsify* command from the terminal and open the **http://0.0.0.0:5543/**. Open the project used to create a new weights file and get the path from the setting link.

# Converting YOLO V3 model to ONNX

* Download the Ultralytics repository (or use the repository if you already have);
* Copy the file `converterYoloV3ToOnnx.py` to Ultralytics local folder;
* Set the variables according to application: weights_file, cfg_file, folder_to_save and new_file_name.  

Run (by terminal) the command `python converterYoloV3ToOnnx.py`. The converted weights will be save on the **folder_to_save** folder.

# Example Model Server and Client Using Flask

The server uses Flask to create an app with the DeepSparse Engine hosting a compiled model as the backend.

The client communicates inputs to the server by serializing NumPy tensors and sending them using the `requests` library.

## Installation

Install DeepSparse with `pip install deepsparse` and the additional external requirements with `pip install -r requirements.txt`.

## Execution

### Server

First, open the `server.py` file and configure the variables: onnx_filepath, address, port and image_size. Start up the host `server.py` with your model of choice.

Example command:
```bash
python server.py
```

You can leave that running as a detached process or in a spare terminal.

This starts a Flask app with the DeepSparse Engine as the inference backend, accessible at `http://0.0.0.0:9898` by default.

The app exposes HTTP endpoints at:
- `/api/info` to get information about the compiled model
- `/api/predict` to send inputs to the model and recieve predicted outputs in response.

### Client

Then, in another terminal, use the `client.py` script to generate inputs, send them to the server's `/api/predict` endpoint, and receive the prediction outputs.

Example command:
```bash
python client.py    
```
