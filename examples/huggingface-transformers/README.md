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

# Hugging Face Transformers - DeepSparse Inference Examples
This directory contains examples for serving, benchmarking, and running NLP 
models from the [Hugging Face Transformers](https://github.com/huggingface/transformers)
repository using the DeepSparse Engine. These examples can load pre-trained,
sparsified models from the [SparseZoo](https://github.com/neuralmagic/sparsezoo) 
or you can specify your own transformer [ONNX](https://onnx.ai/) file.

## Installation
The dependencies for this example can be installed using `pip` and the supplied `requirements.txt` file:
```bash
pip3 install -r requirements.txt
```
## DeepSparse Pipeline Example

The DeepSparse-Hugging Face pipeline integration provides a simple API 
dedicated to several tasks,
following is an example using a pruned BERT model from the SparseZoo for 
Question-Answering task. The current version of the pipeline supports only 
`question-answering` tasks, 

```python
from deepsparse.transformers import pipeline

# SparseZoo model stub or path to ONNX file
onnx_filepath='zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate'

num_cores=None  # uses all available CPU cores by default

# Get DeepSparse question-answering pipeline

qa_pipeline = pipeline(
    task="question-answering",
    model_path=onnx_filepath,
    num_cores=num_cores,
)

# inference

my_name = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```

## Benchmarking Example
`benchmark.py` is a script for benchmarking sparsified Hugging Face Transformers
performance with DeepSparse.  For a full list of options run `python 
benchmark.py -h`.

To run a benchmark using the DeepSparse Engine with a pruned BERT model that uses all available CPU cores and batch size 1, run:
```bash
python benchmark.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate \
    --batch-size 1
```



## Example [Transformers](https://github.com/huggingface/transformers) DeepSparse Flask Deployment

To illustrate how the DeepSparse Engine can be used with Hugging Face 
Transformers, this directory contains a sample model server and client. 

The server uses Flask to create an app with the DeepSparse Engine hosting a
compiled Hugging Face Transformer model.
The client can make requests into the server returning inference results for 
given inputs.

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
- `/predict` to send inputs to the model and receive a response.
    The number of inputs should match the compiled model's batch size.

For a full list of options, run `python server.py -h`.

Currently, the server uses DeepSparse-HuggingFace pipeline integration 
for end-to-end prediction.  

### Client

`client.py` provides a callable `PipelineClient` object to make requests to the 
server easy.
The file is self-documented.  See example usage below:

```python
from client import PipelineClient
remote_model = PipelineClient(address='0.0.0.0', port='5543')
model_outputs = remote_model(question="What's my name?", context="My name is Snorlax")
```

### SparseZoo Stubs


| Model Name     |      Stub      |
|----------|-------------|
| [bert-pruned-moderate]() | zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate |
| [bert-6layers-aggressive-pruned]()| zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_6layers-aggressive_96 |
| [bert-pruned-conservative]()| zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-conservative |
| [pruned_6layers-moderate]() | zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_6layers-moderate |
| [pruned-aggressive_94]() | zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_94|
| [pruned_6layers-conservative]()| zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_6layers-conservative|
| [bert-base]()|zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none |
