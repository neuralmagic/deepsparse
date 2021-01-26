# Example Model Server and Client Using Flask

This directory holds an example server and client implementation. This fulfills the scenario of making inference requests to a real-time server hosting an ONNX model.

The server uses Flask to create an app with the DeepSparse Engine hosting a compiled model as the backend.

The client communicates inputs to the server by serializing NumPy tensors and sending them using the `requests` library.

## Example Execution

First, start up the host server with your model of choice:

```bash
python server.py ~/Downloads/resnet18_pruned.onnx
```

Then, in another terminal, use the client script to generate inputs, send them to the server, and receive the outputs:

```bash
python deepsparse/examples/flask/client.py ~/Downloads/resnet18_pruned.onnx    
[     INFO            onnx.py:  92 - generate_random_inputs() ] Generating 1 random inputs
[     INFO            onnx.py: 102 - generate_random_inputs() ] -- random input #0 of shape = [1, 3, 224, 224]
Sending 1 input tensors to http://0.0.0.0:5543/predict
Recieved response of 2 output tensors:
Round-trip time took 13.4261 milliseconds
    output #0: shape (1, 1000)
    output #1: shape (1, 1000)
```
