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
sparsified models from [SparseZoo](https://github.com/neuralmagic/sparsezoo) 
or you can specify your own transformer [ONNX](https://onnx.ai/) file.

## Installation
The dependencies for these examples can be installed using `pip` and the supplied `requirements.txt` file:
```bash
pip3 install -r requirements.txt
```

## DeepSparse Pipeline Example

The DeepSparse-Hugging Face pipeline integration provides a simple API 
dedicated to several tasks,
following is an example using a pruned BERT model from the SparseZoo for 
Question-Answering task. Other currently supported tasks include
`text-classification`, `sentiment-analysis`, `ner`, and `token-classification`.

```python
from deepsparse.transformers import pipeline

# SparseZoo model stub or path to ONNX file
onnx_filepath="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98"

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
The pipeline can also infer a default sparse model to run on the system.

```python
from deepsparse.transformers import pipeline
qa_pipeline = pipeline("question-answering")
my_name = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```

## Benchmarking Example
`benchmark.py` is a script for benchmarking sparsified Hugging Face Transformers
performance with DeepSparse.  For a full list of options run `python 
benchmark.py -h`.

To run a benchmark using the DeepSparse Engine with a pruned BERT model that uses all available CPU cores and batch size 1, run:
```bash
python benchmark.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98 \
    --batch-size 1
```



## Example Transformers DeepSparse Flask Deployment

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
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98
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
[SparseZoo](http://sparsezoo.neuralmagic.com/) is a constantly-growing repository of sparsified (pruned and 
pruned-quantized) models with matching sparsification recipes for neural networks. It simplifies and accelerates your time-to-value in building performant deep learning models with a collection of inference-optimized models and recipes to prototype from.

Available via API and hosted in the cloud, the [SparseZoo](http://sparsezoo.neuralmagic.com/) contains both 
baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery. Recipe-driven approaches built around sparsification algorithms allow you to take the models as given, transfer-learn from the models onto private datasets, or transfer the recipes to your architectures.

Each model in the SparseZoo has a specific stub that identifies it. The stubs are made up of the following structure:

`DOMAIN/SUB_DOMAIN/ARCHITECTURE{-SUB_ARCHITECTURE}/FRAMEWORK/REPO/DATASET{-TRAINING_SCHEME}/SPARSE_NAME-SPARSE_CATEGORY-{SPARSE_TARGET}`

Learn more at 
- [SparseZoo Docs](https://docs.neuralmagic.com/sparsezoo/)
- [SparseZoo Website](http://sparsezoo.neuralmagic.com/) 
- [SparseZoo GitHub Repository](https://github.com/neuralmagic/sparsezoo)


| Model Name     |      Stub      | Description |
|----------|-------------|-------------|
| bert-6layers-aggressive-pruned-96| zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_6layers-aggressive_96 |This model is the result of pruning a modified BERT base uncased with 6 layers on the SQuAD dataset. The sparsity level is 95% uniformly applied to all encoder layers. Distillation was used with the teacher being the BERT model fine-tuned on the dataset for two epochs.|
| bert-pruned-conservative| zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-conservative |This model is the result of pruning BERT base uncased on the SQuAD dataset. The sparsity level is 80% uniformly applied to all encoder layers. Distillation was used with the teacher being the BERT model fine-tuned on the dataset for two epochs.|
| pruned-aggressive_94 | zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_94|This model is the result of pruning a modified BERT base uncased with 6 layers on the SQuAD dataset. The sparsity level is 90% uniformly applied to all encoder layers. Distillation was used with the teacher being the BERT model fine-tuned on the dataset for two epochs.|
| bert-3layers-pruned-aggressive-89| zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_89|This model is the result of pruning a modified BERT base uncased with 6 layers on the SQuAD dataset. The sparsity level is 89% uniformly applied to all encoder layers. Distillation was used with the teacher being the BERT model fine-tuned on the dataset for two epochs.|
| bert-base|zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none |This model is the result of a BERT base uncased model fine-tuned on the SQuAD dataset for two epochs.|