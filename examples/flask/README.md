<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Example Model Server and Client Using Flask

This directory holds an example server and client implementation. This fulfills the scenario of making inference requests to a real-time server hosting an ONNX model.

The server uses Flask to create an app with the DeepSparse Engine hosting a compiled model as the backend.

The client communicates inputs to the server by serializing NumPy tensors and sending them using the `requests` library.

## Installation

Install DeepSparse with `pip install deepsparse` and the additional external requirements with `pip install -r requirements.txt`.

## Execution

### Server

First, start up the host `server.py` with your model of choice. You can specify `num_cores` to use.

Example command:
```bash
python server.py ~/Downloads/resnet18_pruned.onnx
```

You can leave that running as a detached process or in a spare terminal.

This starts a Flask app with the DeepSparse Engine as the inference backend, accessible at `http://0.0.0.0:5543` by default.

The app exposes HTTP endpoints at:
- `/info` to get information about the compiled model
- `/predict` to send inputs to the model and recieve predicted outputs in response.

### Client

Then, in another terminal, use the `client.py` script to generate inputs, send them to the server's `/predict` endpoint, and receive the prediction outputs.

Example command:
```bash
python client.py ~/Downloads/resnet18_pruned.onnx    
```
Output:
```bash
[     INFO            onnx.py: 127 - generate_random_inputs() ] -- generating random input #0 of shape = [1, 3, 224, 224]
[     INFO          client.py: 152 -                 main() ] Sending 1 input tensors to http://0.0.0.0:5543/run
[    DEBUG          client.py: 102 -                _post() ] Sending POST request to http://0.0.0.0:5543/run
[     INFO          client.py: 159 -                 main() ] Round-trip time took 13.3283 milliseconds
[     INFO          client.py: 160 -                 main() ] Received response of 2 output tensors:
[     INFO          client.py: 163 -                 main() ]   output #0: shape (1, 1000)
[     INFO          client.py: 163 -                 main() ]   output #1: shape (1, 1000)
```
