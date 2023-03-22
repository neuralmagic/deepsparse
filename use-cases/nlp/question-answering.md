# Deploying Question Answering Models with DeepSparse
This page explains how to benchmark and deploy a question answering model with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API. It enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing
and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving
endpoint running DeepSparse with a single CLI.

## Installation Requirements

This use case requires the installation of [DeepSparse Server](/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](/user-guide/deepsparse-engine/hardware-support).

## Benchmarking

We can use the benchmarking utility to demonstrate the DeepSparse's performance. We ran the numbers below on a 23-core server.

### ONNX Runtime Baseline

As a baseline, let's check out ONNX Runtime's performance on BERT. Make sure you have ORT installed (`pip install onnxruntime`).
````bash
deepsparse.benchmark \
  zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none \
  -b 64 -s sync -nstreams 1 -i [64,384] \
  -e onnxruntime
> Original Model Path: zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 9.3279
> Latency Mean (ms/batch): 6861.1296
> Latency Median (ms/batch): 6861.1296
> Latency Std (ms/batch): 876.7494
> Iterations: 2
````

ONNX Runtime achieves 9 items/second with batch 64 and sequence length 384.

## DeepSparse Engine
Now, let's run DeepSparse on an inference-optimized sparse version of BERT. This model has been 90% pruned and quantized, while retaining >99% accuracy of the dense baseline on the [SQuAD](https://huggingface.co/datasets/squad) dataset.
```bash
deepsparse.benchmark \
  zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none \
  -b 64 -s sync -nstreams 1 -i [64,384] \
  -e deepsparse
  
 > Original Model Path: zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 83.3859
> Latency Mean (ms/batch): 767.5012
> Latency Median (ms/batch): 767.5222
> Latency Std (ms/batch): 3.5643
> Iterations: 14
```
DeepSparse achieves 83 items/second, an 9.2x speed-up over ONNX Runtime!
## DeepSparse Engine
Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended using the Pipeline API but Engine is available if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 90% pruned-quantized BERT trained on SQuAD from SparseZoo:
```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none"
batch_size = 1
bert_engine = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)

# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = bert_engine(inputs)
print(output)
#[array([[-6.904723 , -7.2960553, -6.903628 , -6.930577 , -6.899986 ,
# .....
#   -6.555915 , -6.6454444, -6.4477777, -6.8030496]], dtype=float32)]
```
## DeepSparse Pipelines
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of a question answering Pipeline with a 90% pruned-quantized version of BERT trained on SQuAD. We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input) is handled by the `Pipeline`.
```python

from deepsparse import Pipeline
task = "question-answering"
qa_pipeline = Pipeline.create(
        task=task,
        model_path="zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none",
    )
q_context = "DeepSparse is sparsity-aware inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application"
question = "What is DeepSparse?"
output = qa_pipeline(question=question, context=q_context)
print(output)
# QuestionAnsweringOutput(score=23.620140075683594, answer='sparsity-aware inference runtime', start=14, end=46)
```
### Use Case Specific Arguments
The Question Answering Pipeline contains additional arguments for configuring a `Pipeline`.

#### Sequence Length
The `sequence_length` argument adjusts the ONNX graph to handle a specific sequence length. In the DeepSparse Pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance.

The example below compiles the model and runs inference with sequence length 64.
```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none"
batch_size = 1
task = "question-answering"
sequence_length = 64
pipeline = Pipeline.create(
        task=task,
        model_path=sparsezoo_stub,  # sparsezoo stub or path to local ONNX
    sequence_length=sequence_length,
    batch_size =batch_size, # default batch size is 1
    )

# run inference on image file
q_context = "DeepSparse is sparsity-aware inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application"
question = "What is DeepSparse?"
output = qa_pipeline(question=question, context=q_context)
print(output)
# QuestionAnsweringOutput(score=23.620140075683594, answer='sparsity-aware inference runtime', start=14, end=46)
```
Alternatively, you can pass a list of sequence lengths, creating a "bucketable" pipeline. Under the hood, the DeepSparse Pipeline will compile multiple versions of the engine (utilizing a shared scheduler) and direct inputs towards the smallest bucket into which it fits.

The example below creates a bucket for smaller input lengths (16 tokens) and for larger input lengths (128 tokens).
```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none"
batch_size = 1
task = "question-answering"
buckets = [16, 128]

pipeline = Pipeline.create(
        task=task,
        model_path=sparsezoo_stub,  # sparsezoo stub or path to local ONNX
    sequence_length=buckets, # creates bucketed pipeline
    batch_size =batch_size,  # default batch size is 1
    )

# run inference on image file
q_context = "DeepSparse is sparsity-aware inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application"
question = "What is DeepSparse?"
output = qa_pipeline(question=question, context=q_context)
print(output)
# QuestionAnsweringOutput(score=23.620140075683594, answer='sparsity-aware inference runtime', start=14, end=46)
```
### Cross Use Case Functionality
Check out the [Pipeline User Guide](/user-guide/deepsparse/deepsparse-pipelines) for more details on configuring a Pipeline.
## DeepSparse Server
DeepSparse Server is built on top of FastAPI and Uvicorn, enabling you to set up a REST endpoint for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all the utilities provided by Pipelines.

The CLI command below launches a question answering pipeline with a 90% pruned-quantized BERT model:

```bash
deepsparse.server \
  --task question-answering \
  --model_path "zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none" # or path/to/onnx
```
You should see Uvicorn report that it is running on http://0.0.0.0:5543. Once launched, a /docs path is created with full endpoint descriptions and support for making sample requests.

Here is an example client request, using the Python requests library for formatting the HTTP:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {
  "question": "What is DeepSparse?",
  "context": "DeepSparse is sparsity-aware inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application",
}

resp = requests.post(url=url, json=obj)

# receive the post-processed output
print(resp.text)
# {"score":23.620140075683594,"answer":"sparsity-aware inference runtime","start":14,"end":46}
```
#### Use Case Specific Arguments
To use the `sequence_length` argument, create a server configuration file for passing the arguments via `kwargs`.

This configuration file sets sequence length to 64:
```yaml
# question-answering-config.yaml
endpoints:
  - task: question-answering
    model: zoo:nlp/question_answering/bert-base_cased/pytorch/huggingface/squad/pruned90_quant-none
    kwargs:
      sequence_length: 64       # uses sequence length 64
```
Spin up the server:

```bash
deepsparse.server \
  --config-file question-answering-config.yaml
```
Making a request:
```python
import requests

# Uvicorn is running on this port
url = "http://localhost:5543/predict"

# send the data
obj = {
    "question": "Who is Mark?",
    "context": "Mark is batman."
}

response = requests.post(url, json=obj)
# receive the post-processed output
print(response.text)
# {"score":22.506305694580078,"answer":"batman","start":8,"end":14}
```
### Cross Use Case Functionality

Check out the [Server User Guide](/user-guide/deepsparse/deepsparse-server) for more details on configuring the Server.
