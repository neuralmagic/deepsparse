# Deploying Text Classification Models with DeepSparse

This page explains how to benchmark and deploy a text classification model with DeepSparse.

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

As a baseline, let's check out ONNX Runtime's performance on oBERT. Make sure you have ORT installed (`pip install onnxruntime`).
```bash
deepsparse.benchmark \
  zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none \
  -b 64 -s sync -nstreams 1 -i [64,384] \
  -e onnxruntime

> Original Model Path: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 10.7702
> Latency Mean (ms/batch): 5942.3135
> Latency Median (ms/batch): 5942.3135
> Latency Std (ms/batch): 309.5893
> Iterations: 2
```
ONNX Runtime achieves 11 items/second with batch 64 and sequence length 384.

### DeepSparse Speedup
Now, let's run DeepSparse on an inference-optimized sparse version of oBERT. This model has been 90% pruned and quantized, while retaining >99% accuracy of the dense baseline on the MNLI dataset.
```bash
!deepsparse.benchmark \
  zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none \
  -b 64 -s sync -nstreams 1 -i [64,384] \
  -e deepsparse

> Original Model Path: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 84.7056
> Latency Mean (ms/batch): 755.5426
> Latency Median (ms/batch): 758.5148
> Latency Std (ms/batch): 5.9118
> Iterations: 14
```
DeepSparse achieves 85 items/second, an 7.7x speed-up over ONNX Runtime!

## DeepSparse Engine
Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended you use the Pipeline API but Engine is available as needed if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 90% pruned-quantized oBERT trained on MNLI from SparseZoo:
```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
batch_size = 1
bert_engine = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)

# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = bert_engine(inputs)
print(output)
# [array([[-0.9264987, -1.6990623,  2.3935342]], dtype=float32)]

```
## DeepSparse Pipelines
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of a text classification Pipeline with a 90% pruned-quantized version of oBERT trained on MNLI. We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input) is handled by the `Pipeline`.
```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
batch_size = 1
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1                 # default batch size is 1
)

# run inference on image file
prediction = pipeline("The text classification pipeline is fast and easy to use")
print(prediction)
# labels=['entailment'] scores=[0.5807693004608154]

```
#### Zero Shot Classification
Given certain categories, zero shot classification aims at determining the class that best fits the given text.

Here's an example of a zero shot text classification example with a DistilBERT that's 80% pruned and quantized on the MNLI dataset.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/mnli/pruned80_quant-none-vnni"
batch_size = 1
pipeline = Pipeline.create(
        task="zero_shot_text_classification",
        model_path="zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/mnli/pruned80_quant-none-vnni",
    )

# run inference on image file
prediction = pipeline(sequences = "Today is election day in America",labels=['politics', 'public health', 'Europe'],)
print(prediction)
# sequences='Today is election day in America' labels=['politics', 'Europe', 'public health'] scores=[0.8922364115715027, 0.06215662881731987, 0.04560691863298416]
```
### Example with QQP
[QQP( Quora Question Pairs2)](https://huggingface.co/datasets/glue) is a dataset that is part of the GLUE benchmark. The goal is to determine if a pair of questions are semantically equivalent.

Let's illustrate that using a SparseZoo `obert` model that has been pruned to 90%(90% of the weights have been removed without loss of accuracy) and quantized on the QQP dataset.
```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/qqp/pruned90_quant-none"
batch_size = 1
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1                 # default batch size is 1
)

# run inference on image file
prediction = pipeline([[
        "Should I have a hair transplant at age 24? How much would it cost?",
        "How much cost does hair transplant require?",
    ]])
print(prediction)
# labels=['not_duplicate'] scores=[0.6760590076446533]

```
### Example with MNLI
[MNLI( Multi-Genre Natural Language Inference)](https://huggingface.co/datasets/glue) is a dataset with textual entailment annotations. The goal is to predict entailment, contradiction and neutrality given a premise and hypothesis.

Let's illustrate that using a SparseZoo `obert` model that has been pruned to 90%(90% of the weights have been removed without loss of accuracy) and quantized on the MNLI dataset.
```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
batch_size = 1
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1                 # default batch size is 1
)

# run inference on image file
prediction = pipeline([[
        "Timely access to information is in the best interests of both GAO and the agencies",
        "It is in everyone's best interest to have access to information in a timely manner",
    ]])
print(prediction)
# labels=['entailment'] scores=[0.9688315987586975]

```

### Example with Document Classification
Document classification involves classifying text in a long document.

Let's illustrate that using a SparseZoo `obert` model that has been pruned to 90% and quantized on the IMDB dataset.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/pruned90_quant-none"
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1                 # default batch size is 1
)

# run inference on image file
prediction = pipeline(["Phil the Alien is one of those quirky films where the humour is based around the oddness of everything rather than actual punchlines.<br /><br />At first it was very odd and pretty funny but as the movie progressed I didn't find the jokes or oddness funny anymore.<br /><br />Its a low budget film (thats never a problem in itself), there were some pretty interesting characters, but eventually I just lost interest.<br /><br />I imagine this film would appeal to a stoner who is currently partaking.<br /><br />For something similar but better try Brother from another planet"])
print(prediction)
# labels=['0'] scores=[0.9986200332641602]


```
### Example with GoEmotions
[The GoEmotions](https://huggingface.co/datasets/go_emotions) dataset contains Reddit comments labeled for 27 emotion categories or Neutral. The goal is to perform multi-class, multi-label emotion classification.

Let's illustrate that using a SparseZoo `obert` model that has been pruned to 90% and quantized on the GoEmotions dataset.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/multilabel_text_classification/obert-base/pytorch/huggingface/goemotions/pruned90-none"
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1                 # default batch size is 1
)

# run inference on image file
prediction = pipeline(["Thank you for asking questions and recognizing that there may be things that you don’t know or understand about police tactics. Seriously. Thank you."])
print(prediction)
#labels=['gratitude'] scores=[0.9986923336982727]

```
### Use Case Specific Arguments
The Text Classification Pipeline contains additional arguments for configuring a `Pipeline`.

#### Sequence Length
The `sequence_length` argument adjusts the ONNX graph to handle a specific sequence length. In the DeepSparse Pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance.

The example below compiles the model and runs inference with sequence length 64.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/mnli/pruned80_quant-none-vnni"
batch_size = 1
sequence_length = 64
pipeline = Pipeline.create(
        task="zero_shot_text_classification",
        model_path=sparsezoo_stub,
    sequence_length=sequence_length,
    batch_size =batch_size,
    )

# run inference on image file
prediction = pipeline(sequences = "Today is election day",labels=['politics', 'public health', 'Europe'],)
print(prediction)
# sequences='Today is election day' labels=['politics', 'Europe', 'public health'] scores=[0.9697986245155334, 0.01720993034541607, 0.012991504743695259]
```
Alternatively, you can pass a list of sequence lengths, creating a "bucketable" pipeline. Under the hood, the DeepSparse Pipeline will compile multiple versions of the engine (utilizing a shared scheduler) and direct inputs towards the smallest bucket into which it fits.

The example below creates a bucket for smaller input lengths (16 tokens) and for larger input lengths (128 tokens).
```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
batch_size = 1
buckets = [16, 128]
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1,
  sequence_length=buckets  # creates bucketed pipeline
)
# run inference on image file
prediction = pipeline([[
        "Timely access to information is in the best interests of both GAO and the agencies",
        "It is in everyone's best interest to have access to information in a timely manner",
    ]])
print(prediction)

# run inference on image file
prediction = pipeline([[
        "Timely access to information is in the best interests of both GAO and the agencies. Let's make information more accessible",
        "It is in everyone's best interest to have access to information in a timely manner. Information should be made more accessible.",
    ]])
print(prediction)
#labels=['entailment'] scores=[0.9688315987586975]
#labels=['entailment'] scores=[0.985545814037323]

```

#### Return All Scores
The `return_all_scores` argument allows you to specify whether to return the prediction as the `argmax` of class predictions or to return all scores as a list for each result in the batch.

Here is an example with batch size 1 and batch size 2:
```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
pipeline_b1 = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,    # sparsezoo stub or path to local ONNX
  batch_size=1,                 # default batch size is 1
  return_all_scores=True        # default is false
)

# download onnx from sparsezoo and compile with batch size 2
pipeline_b2 = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,    # sparsezoo stub or path to local ONNX
  batch_size=2,        # default batch size is 1
  return_all_scores=True        # default is false
)

# run inference with b1
sequences_b1 = [[
        "Timely access to information is in the best interests of both GAO and the agencies",
        "It is in everyone's best interest to have access to information in a timely manner",
    ]]
prediction_b1 = pipeline_b1(sequences_b1)
print(prediction_b1)

# run inference with b2
sequences_b2 = sequences_b1 * batch_size
prediction_b2 = pipeline_b2(sequences_b2)
print(prediction_b2)
# labels=[['entailment', 'neutral', 'contradiction']] scores=[[0.9688315987586975, 0.030656637623906136, 0.0005117706023156643]]
# labels=[['entailment', 'neutral', 'contradiction'], ['entailment', 'neutral', 'contradiction']] scores=[[0.9688315987586975, 0.030656637623906136, 0.0005117706023156643], [0.9688315987586975, 0.030656637623906136, 0.0005117706023156643]]
```
### Cross Use Case Functionality
Check out the [Pipeline User Guide](/user-guide/deepsparse/deepsparse-pipelines) for more details on configuring a Pipeline.
## DeepSparse Server
Built on the popular FastAPI and Uvicorn stack, DeepSparse Server enables you to set up a REST endpoint for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all the utilities provided by Pipelines.

The CLI command below launches a text classification pipeline with a 90% pruned-quantized oBERT model:

```bash
deepsparse.server \
  --task sentiment-analysis \
  --model_path "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none" # or path/to/onnx

```
You should see Uvicorn report that it is running on http://0.0.0.0:5543. Once launched, a /docs path is created with full endpoint descriptions and support for making sample requests.

Here is an example client request, using the Python requests library for formatting the HTTP:

```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {"sequences": "Sending requests to DeepSparse Server is fast and easy!"}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# {"labels":["entailment"],"scores":[0.5475465655326843]}

```
### Use Case Specific Arguments
To use the `sequence_length` and `return_all_scores` arguments, create a Server configuration file for passing the arguments via kwargs.

This configuration file sets sequence length to 64 and returns all scores:
```yaml
  # text-classification-config.yaml
endpoints:
  - task: text-classification
    model: zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/pruned90_quant-none
    kwargs:
      sequence_length: 64       # uses sequence length 64
      return_all_scores: True   # returns all scores
```
Spin up the server:

```bash

deepsparse.server \
  --config-filetext-classification-config.yaml
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
# {"labels":[["1","0"]],"scores":[[0.9941965341567993,0.005803497973829508]]}

```
### Cross Use Case Functionality

Check out the [Server User Guide](/user-guide/deepsparse/deepsparse-server) for more details on configuring the Server.
