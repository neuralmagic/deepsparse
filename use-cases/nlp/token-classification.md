# Deploying Token Classification Models with DeepSparse

This page explains how to benchmark and deploy a token classification model with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing
and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving
endpoint running DeepSparse with a single CLI.

## Installation Requirements

This use case requires the installation of [DeepSparse Server](https://docs.neuralmagic.com/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](https://docs.neuralmagic.com/user-guides/deepsparse-engine/hardware-support).

## Benchmarking

We can use the benchmarking utility to demonstrate the DeepSparse's performance. We ran the numbers below on a 4 core AWS `c6i.2xlarge` instance.

### ONNX Runtime Baseline
As a baseline, let's check out ONNX Runtime's performance on BERT. Make sure you have ORT installed (`pip install onnxruntime`).

```bash
deepsparse.benchmark \
  zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none \
  -b 64 -s sync -nstreams 1 -i [64,128] \
  -e onnxruntime

> Original Model Path: zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 19.96
```
ONNX Runtime achieves 20 items/second with batch 64 and sequence length 128.

## DeepSparse Speedup

Now, let's run DeepSparse on an inference-optimized sparse version of BERT. This model has been 90% pruned and quantized, while retaining >99% accuracy of the dense baseline on the conll dataset.

```bash
deepsparse.benchmark \
  zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none \
  -b 64 -s sync -nstreams 1 -i [64,128] \
  -e deepsparse

> Original Model Path: Original Model Path: zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 126.5129
```

DeepSparse achieves 127 items/second, a 6.4x speed-up over ONNX Runtime!

## DeepSparse Engine

Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended using the Pipeline API but Engine is available if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 80% pruned-quantized BERT trained on conll2003 from SparseZoo:
```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
batch_size = 1
compiled_model = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)

# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = compiled_model(inputs)
print(output)
# array([[[ 2.0983224 ,  1.2409506 , -1.7314302 , ..., -0.07210742,
#...
#  -2.0502508 , -2.956191  ]]], dtype=float32)]
```

## DeepSparse Pipelines
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of a token classification Pipeline with a 90% pruned-quantized version of BERT trained on conll2003. We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input) is handled by the `Pipeline`.

```python
from deepsparse import Pipeline
model_path = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
pipeline = Pipeline.create(
        task="token_classification",
        model_path=model_path,
    )
output = pipeline("Mary is flying from Nairobi to New York")
print(output.predictions)
# [[TokenClassificationResult(entity='B-PER', score=0.9971914291381836, word='mary', start=0, end=4, index=1, is_grouped=False),
#  TokenClassificationResult(entity='B-LOC', score=0.9993892312049866, word='nairobi', start=20, end=27, index=5, is_grouped=False),
#  TokenClassificationResult(entity='B-LOC', score=0.9993736147880554, word='new', start=31, end=34, index=7, is_grouped=False),
#  TokenClassificationResult(entity='I-LOC', score=0.997299075126648, word='york', start=35, end=39, index=8, is_grouped=False)]]
```

### Use Case Specific Arguments
The Token Classification Pipeline contains additional arguments for configuring a `Pipeline`.

#### Sequence Length
The `sequence_length` argument adjusts the ONNX graph to handle a specific sequence length. In the DeepSparse Pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance.

The example below compiles the model and runs inference with sequence length of 64.
```python
from deepsparse import Pipeline
model_path = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
pipeline = Pipeline.create(
        task="token_classification",
        model_path=model_path,
        sequence_length=64
    )
print(output.predictions)
# [[TokenClassificationResult(entity='B-PER', score=0.9971914291381836, word='mary', start=0, end=4, index=1, is_grouped=False),
#  TokenClassificationResult(entity='B-LOC', score=0.9993892312049866, word='nairobi', start=20, end=27, index=5, is_grouped=False),
#  TokenClassificationResult(entity='B-LOC', score=0.9993736147880554, word='new', start=31, end=34, index=7, is_grouped=False),
#  TokenClassificationResult(entity='I-LOC', score=0.997299075126648, word='york', start=35, end=39, index=8, is_grouped=False)]]
```

Alternatively, you can pass a list of sequence lengths, creating a "bucketable" pipeline. Under the hood, the DeepSparse Pipeline will compile multiple versions of the model (utilizing a shared scheduler) and direct inputs towards the smallest bucket into which it fits.

The example below creates a bucket for smaller input lengths (64 tokens) and for larger input lengths (128 tokens).
```python
from deepsparse import Pipeline, Context
model_path = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
pipeline = Pipeline.create(
        task="token_classification",
        model_path=model_path,
        sequence_length = [64,128],
        context=Context(num_streams=1)
    )
output = pipeline("Mary is flying from Nairobi to New York to attend a conference")
print(output.predictions)
# [[TokenClassificationResult(entity='B-PER', score=0.9971914291381836, word='mary', start=0, end=4, index=1, is_grouped=False),
#  TokenClassificationResult(entity='B-LOC', score=0.9993892312049866, word='nairobi', start=20, end=27, index=5, is_grouped=False),
#  TokenClassificationResult(entity='B-LOC', score=0.9993736147880554, word='new', start=31, end=34, index=7, is_grouped=False),
#  TokenClassificationResult(entity='I-LOC', score=0.997299075126648, word='york', start=35, end=39, index=8, is_grouped=False)]]
```

#### Aggregation Strategy

`aggregation_strategy` specifies how to aggregate tokens in post-processing in a case where a single word is split into multiple tokens by the tokenizer. The default is to use `none`, which means that we perform no aggregation.

Here is an example using `simple` aggregation strategy.

```python
from deepsparse import Pipeline
model_path = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
pipeline = Pipeline.create(
    task="token_classification",
    model_path=model_path,
    aggregation_strategy="simple"
)

output = pipeline("The Uzbekistani striker scored a goal in the final minute to defeat the Italian national team")
print(output.predictions)

# [[TokenClassificationResult(entity='MISC', score=0.9935868382453918, word='uzbekistani', start=4, end=15, index=None, is_grouped=True),
#  TokenClassificationResult(entity='MISC', score=0.9991180896759033, word='italian', start=72, end=79, index=None, is_grouped=True)]]
```

In comparison, here is the standard output withe no aggregation:

```python
from deepsparse import Pipeline
model_path = "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none"
pipeline = Pipeline.create(
    task="token_classification",
    model_path=model_path,
    aggregation_strategy="none"
)

output = pipeline("The Uzbekistani striker scored a goal in the final minute to defeat the Italian national team")
print(output.predictions)

# [[[TokenClassificationResult(entity='B-MISC', score=0.9973152279853821, word='uzbekistan', start=4, end=14, index=2, is_grouped=False),
#  TokenClassificationResult(entity='I-MISC', score=0.9898584485054016, word='##i', start=14, end=15, index=3, is_grouped=False),
#  TokenClassificationResult(entity='B-MISC', score=0.9991180896759033, word='italian', start=72, end=79, index=15, is_grouped=False)]]
```

### Cross Use Case Functionality
Check out the Pipeline User Guide for more details on configuring a Pipeline.

## DeepSparse Server

DeepSparse Server is built on top of FastAPI and Uvicorn, enabling you to set up a REST endpoint for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all the utilities provided by Pipelines.

The CLI command below launches a token classification pipeline with a 90% pruned-quantized BERT model trained on Conll2003:

```bash 
deepsparse.server \
  --task token_classification \
  --model_path "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none" # or path/to/onnx
```
You should see Uvicorn report that it is running on http://0.0.0.0:5543. Once launched, a /docs path is created with full endpoint descriptions and support for making sample requests.

Here is an example client request, using the Python requests library for formatting the HTTP:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'
# send the data
obj = {"inputs": "Mary is flying from Nairobi to New York to attend a conference"}
resp = requests.post(url=url, json=obj)
# receive the post-processed output
print(resp.text)
# {"predictions":[[{"entity":"B-PER","score":0.9966245293617249,"word":"mary","start":0,"end":4,"index":1,"is_grouped":false},{"entity":"B-LOC","score":0.999544084072113,"word":"nairobi","start":20,"end":27,"index":5,"is_grouped":false},{"entity":"B-LOC","score":0.9993794560432434,"word":"new","start":31,"end":34,"index":7,"is_grouped":false},{"entity":"I-LOC","score":0.9970214366912842,"word":"york","start":35,"end":39,"index":8,"is_grouped":false}]]}
```

#### Use Case Specific Arguments
To use the `sequence_length` and `aggregation_strategy` arguments, create a server configuration file for passing the arguments via `kwargs`.

This configuration file sets sequence length to 64 with `simple` aggregation strategy:
```yaml
# ner-config.yaml
endpoints:
  - task: token_classification
    model: zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none
    kwargs:
      sequence_length: 64       # uses sequence length 64
      aggregation_strategy: simple
```
Spin up the server:

```bash
deepsparse.server \
  --config-file ner-config.yaml
```
Making a request:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {"inputs": "Mary is flying from Nairobi to New York to attend a conference",}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# {"predictions":[[{"entity":"PER","score":0.9966245293617249,"word":"mary","start":0,"end":4,"index":null,"is_grouped":true},{"entity":"LOC","score":0.999544084072113,"word":"nairobi","start":20,"end":27,"index":null,"is_grouped":true},{"entity":"LOC","score":0.9982004165649414,"word":"new york","start":31,"end":39,"index":null,"is_grouped":true}]]}
```

### Cross Use Case Functionality

Check out the Server User Guide for more details on configuring the Server.