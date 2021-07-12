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

# Hugging Face transformers - DeepSparse Inference Examples
This directory contains examples of benchmarking, and serving Hugging Face transformers from the [hugging-face-transformers](https://github.com/huggingface/transformers)
repository using the DeepSparse Engine. These examples can load pre-trained,
sparsified models from [SparseZoo](https://github.com/neuralmagic/sparsezoo) 
or you can specify your own transformer ONNX file

## Installation
The dependencies for this example can be installed using `pip`:
```bash
pip3 install -r requirements.txt
```

## Benchmarking Example
`benchmark.py` is a script for benchmarking sparsified and quantized 
hugging face transformers
performance with DeepSparse.  For a full list of options run `python 
benchmark.py -h`.

To run a benchmark run:
```bash
python benchmark.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate \
    --batch-size 1
```



# Example [transformer](https://github.com/huggingface/transformers) DeepSparse Flask Server

To illustrate how the DeepSparse Engine can be used with Hugging Face 
transformers, this directory contains a sample model server and client. 

The server uses Flask to create an app with the DeepSparse Engine hosting a
compiled Hugging Face transformer model.
The client can make requests into the server returning object detection results for given images.

### Server

First, start up the host `server.py` with your model of choice, SparseZoo stubs are
also supported.

Example command:
```bash
python server.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate
```

You can leave that running as a detached process or in a spare terminal.

This starts a Flask app with the DeepSparse Engine as the inference backend, accessible at `http://0.0.0.0:5543` by default.

The app exposes HTTP endpoints at:
- `/info` to get information about the compiled model
- `/predict` to send images to the model and receive as detected in response.
    The number of images should match the compiled model's batch size.

For a full list of options, run `python server.py -h`.

Currently, the server uses deepsparse-huggingface pipeline integration 
for end to end prediction.  

### Client

`client.py` provides a Callable `PipelineClient` object to make requests to the 
server easy.
The file is self-documented.  See example usage below:

```python
from client import PipelineClient
remote_model = PipelineClient()
model_outputs = remote_model(question="What's my name?", context="My name is Snorlax")
```
