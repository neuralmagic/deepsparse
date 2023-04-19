# Deploying Sentiment Analysis Models with DeepSparse

This page explains how to benchmark and deploy a sentiment analysis model with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API. It enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similiar in concept to Hugging Face Pipelines, it wraps Engine with pre-preprocessing and post-processing, allowing you to make requests on raw data and recieve post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on FastAPI and Uvicorn. It enables you to stand up a model serving endpoint running DeepSparse with a single CLI.

## Installation Requirements

This use case requires the installation of [DeepSparse Server](https://docs.neuralmagic.com/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](https://docs.neuralmagic.com/user-guides/deepsparse-engine/hardware-support).

## Benchmarking

We can use the benchmarking utility to demonstrate the DeepSparse's performance. We ran the numbers below on a 4 core AWS `c6i.2xlarge` instance.

### ONNX Runtime Baseline

As a baseline, let's check out ONNX Runtime's performance on BERT. Make sure you have ORT installed (`pip install onnxruntime`).

```bash
deepsparse.benchmark \
  zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none \
  -b 64 -s sync -nstreams 1 \
  -e onnxruntime

> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 19.61
```

ONNX Runtime achieves 20 items/second with batch 64 and sequence length 128.

### DeepSparse Speedup

Now, let's run DeepSparse on an inference-optimized sparse version of BERT. This model has been 90% pruned and quantized, while
retaining >99% accuracy of the dense baseline on the SST2 dataset.

```bash
deepsparse.benchmark \
  zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none \
  -b 64 -s sync -nstreams 1 \
  -e deepsparse

> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 125.80
```

DeepSparse achieves 126 items/second, an 6.4x speed-up over ONNX Runtime!

## DeepSparse Engine

Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended you use the Pipeline API but Engine is available as needed if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 90% pruned-quantized BERT trained on SST2 from SparseZoo:

```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
batch_size = 1
compiled_model = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)

# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = compiled_model(inputs)
print(output)

# >> [array([[-0.3380675 ,  0.09602544]], dtype=float32)]
```

## DeepSparse Pipelines

Pipeline is the default interface for interacting with DeepSparse. 

Just like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine.
This creates a clean API that allows you to pass raw images to DeepSparse and recieve back the post-processed prediction,
making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of a sentiment analysis Pipeline
with a 90% pruned-quantized version of BERT trained on SST2. We can then pass the Pipeline raw text and recieve the predictions. 
All of the pre-processing (such as tokenizing the input) is handled by the Pipeline.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
batch_size = 1
sa_pipeline = Pipeline.create(
  task="sentiment-analysis",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1                 # default batch size is 1
)

# run inference
prediction = sa_pipeline("The sentiment analysis pipeline is fast and easy to use")
print(prediction)

# >>> labels=['positive'] scores=[0.9955807328224182]
```

### Use Case Specific Arguments

The Sentiment Analysis Pipeline contains additional arguments for configuring a Pipeline.

#### Sequence Length

DeepSparse uses static input shapes. We can use the `sequence_length` argument to adjust the ONNX graph to handle a specific sequence length. Inside the DeepSparse pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance.

The example below compiles the model and runs inference with sequence length 64.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
batch_size = 1
sequence_length = 64
sa_pipeline = Pipeline.create(
  task="sentiment-analysis",
  model_path=sparsezoo_stub,    # sparsezoo stub or path to local ONNX
  batch_size=1,                 # default batch size is 1
  sequence_length=64            # default sequence length is 128
)

# run inference on image file
prediction = sa_pipeline("The sentiment analysis pipeline is fast and easy to use")
print(prediction)

# >>> labels=['positive'] scores=[0.9955807328224182]
```

If your input data has a variable distribution of seuqence lengths, you can simulate dynamic shape infernece by passing a list of sequence lengths to DeepSparse, which a "bucketable" pipeline. Under the hood, the DeepSparse Pipeline compile multiple versions of the model at each sequence length (utilizing a shared scheduler) and directs inputs towards the smallest bucket into which it fits.

The example below creates a bucket for smaller input lengths (16 tokens) and for larger input lengths (128 tokens).

```python
from deepsparse import Pipeline, Context

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
batch_size = 1
buckets = [16, 128]
sa_pipeline = Pipeline.create(
  task="sentiment-analysis",
  model_path=sparsezoo_stub,        # sparsezoo stub or path to local ONNX
  batch_size=1,                     # default batch size is 1
  sequence_length=buckets,          # creates bucketed pipeline
  context = Context(num_streams=1)  # creates scheduler with one stream
)

# run inference on short sequence
prediction = sa_pipeline("I love short sentences!")
print(prediction)

# run inference on long sequence
prediction = sa_pipeline("Normal sized sequences take a lot longer to run but are I still like them a lot because of the speedup from DeepSparse")
print(prediction)

# >>> labels=['positive'] scores=[0.9988369941711426]
# >>> labels=['positive'] scores=[0.9587154388427734]
```

#### Return All Scores

The `return_all_scores` argument allows you to specify whether to return the prediction as the argmax of class predictions or
to return all scores as a list for each result in the batch.

Here is an example with batch size 1 and batch size 2:

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
sa_pipeline_b1 = Pipeline.create(
  task="sentiment-analysis",
  model_path=sparsezoo_stub,    # sparsezoo stub or path to local ONNX
  batch_size=1,                 # default batch size is 1
  return_all_scores=True        # default is false
)

# download onnx from sparsezoo and compile with batch size 2
batch_size = 2
sa_pipeline_b2 = Pipeline.create(
  task="sentiment-analysis",
  model_path=sparsezoo_stub,    # sparsezoo stub or path to local ONNX
  batch_size=batch_size,        # default batch size is 1
  return_all_scores=True        # default is false
)

# run inference with b1
sequences_b1 = ["Returning all scores is a cool configuration option"]
prediction_b1 = sa_pipeline_b1(sequences_b1)
print(prediction_b1)

# run inference with b2
sequences_b2 = sequences_b1 * batch_size
prediction_b2 = sa_pipeline_b2(sequences_b2)
print(prediction_b2)

# >>> labels=[['positive', 'negative']] scores=[[0.9845395088195801, 0.015460520051419735]]
# >>> labels=[['positive', 'negative'], ['positive', 'negative']] scores=[[0.9845395088195801, 0.015460520051419735], [0.9845395088195801, 0.015460520051419735]]
```

### Cross Use Case Functionality

Check out the Pipeline User Guide for more details on configuring a Pipeline.

## DeepSparse Server

Built on the popular FastAPI and Uvicorn stack, DeepSparse Server enables you to set-up a REST endpoint 
for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it
inherits all of the utilities provided by Pipelines.

The CLI command below launches an sentiment analysis pipeline with a 90% pruned-quantized 
BERT model identifed by its SparseZoo stub:

```bash
deepsparse.server \
  --task sentiment-analysis \
  --model_path "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none" # or path/to/onnx
```

You should see Uvicorn report that it is running on `http://0.0.0.0:5543`. Once launched, a `/docs` path is 
created with full endpoint descriptions and support for making sample requests.

Here is an example client request, using the Python `requests` library for formatting the HTTP:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {"sequences": "Sending requests to DeepSparse Server is fast and easy!"}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# >> {"labels":["positive"],"scores":[0.9330279231071472]}
```

### Use Case Specific Arguments

To use the `sequence_length` and `return_all_scores` arguments, we can a Server configuration file, passing the arguments via `kwargs`

This configuration file sets sequence length to 64 and returns all scores:

```yaml
# sentiment-analysis-config.yaml
endpoints:
  - task: sentiment-analysis
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
    kwargs:
      sequence_length: 64       # uses sequence length 64
      return_all_scores: True   # returns all scores
```

Spinning up:
```bash
deepsparse.server \
  --config-file sentiment-analysis-config.yaml
```

Making a request:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {"sequences": "Sending requests to DeepSparse Server is fast and easy!"}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# >> {"labels":[["1","0"]],"scores":[[0.9941965341567993,0.005803497973829508]]}
```

### Cross Use Case Functionality

Check out the Server User Guide for more details on configuring the Server.
