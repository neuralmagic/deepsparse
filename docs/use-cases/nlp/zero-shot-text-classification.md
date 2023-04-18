# Deploying Zero Shot Text Classification Models

This page explains how to benchmark and deploy a zero-shot text classification model with DeepSparse.


There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API. It enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing
and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving
endpoint running DeepSparse with a single CLI.

We will walk through an example of each.

## Installation Requirements

This use case requires the installation of [DeepSparse Server](https://docs.neuralmagic.com/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](https://docs.neuralmagic.com/user-guides/deepsparse-engine/hardware-support).

## Task Overview

Zero-shot text classification allows us to perform text classification over any set of potential labels without training a text classification model on those specific labels. 

We can accomplish this goal via two steps:
- Train a model to predict whether a given pair of sentences is an `entailment`, `neutral`, or `contradiction` (on a dataset like MNLI)
- For each sentence `S` and set of labels `L`, predict label `L_i` which has the highest entailment score between `S` and a hypothesis of the form `This text is related to {L_i}` as predicted by the model.

## DeepSparse Pipelines

Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of a zero-shot text classification Pipeline with a 90% pruned-quantized version of oBERT trained on MNLI. We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input and formatting the hypothesis) is handled by the `Pipeline`.

Here's an example with a single input and single label prediction with a model trained on MNLI:

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
batch_size = 1
pipeline = Pipeline.create(
  task="zero_shot_text_classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1,                # default batch size is 1
  labels=["poltics", "public health", "sports"]
)

# run inference
prediction = pipeline("Who are you voting for in the upcoming election")
print(prediction)

# sequences='Who are you voting for in the upcoming election' labels=['poltics', 'sports', 'public health'] scores=[0.5765101909637451, 0.23050746321678162, 0.19298239052295685]
```

We can also pass the labels at inference time:

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
pipeline = Pipeline.create(
  task="zero_shot_text_classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1,                # default batch size is 1
)

# run inference
prediction = pipeline(
    sequences="My favorite sports team is the Boston Red Sox",
    labels=["sports", "politics", "public health"]
)
print(prediction)

# sequences='My favorite sports team is the Boston Red Sox' labels=['sports', 'politics', 'public health'] scores=[0.9349604249000549, 0.048094600439071655, 0.016944952309131622]
```

### Use Case Specific Arguments
The Zero Shot Text Classification Pipeline contains additional arguments for configuring a `Pipeline`.

#### Sequence Length
The `sequence_length` argument adjusts the ONNX graph to handle a specific sequence length. In the DeepSparse Pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance. The default sequence length for text classification is 128.

The example below runs the zero-shot text classification at sequence length 64.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
pipeline = Pipeline.create(
  task="zero_shot_text_classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1,                # default batch size is 1
  sequence_length=64
)

# run inference
prediction = pipeline(
    sequences="My favorite sports team is the Boston Red Sox",
    labels=["sports", "politics", "public health"]
)
print(prediction)

# sequences='My favorite sports team is the Boston Red Sox' labels=['sports', 'politics', 'public health'] scores=[0.9349604249000549, 0.048094600439071655, 0.016944952309131622]
```

Alternatively, you can pass a list of sequence lengths, creating a "bucketable" pipeline. Under the DeepSparse Pipeline will compile multiple versions of the model (utilizing a shared scheduler) and direct inputs towards the smallest bucket into which an input fits.

The example below creates a bucket for smaller input lengths (16 tokens) and for larger input lengths (128 tokens).
```python
from deepsparse import Pipeline, Context

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
pipeline = Pipeline.create(
  task="zero_shot_text_classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1,                # default batch size is 1
  sequence_length=[32,128],
  context=Context(num_streams=1)
)

# run inference
prediction = pipeline(
    sequences="My favorite sports team is the Boston Red Sox",
    labels=["sports", "politics", "public health"]
)
print(prediction)

# sequences='My favorite sports team is the Boston Red Sox' labels=['sports', 'politics', 'public health'] scores=[0.9349604249000549, 0.048094600439071655, 0.016944952309131622]
```

### Model Config

Additionally, we can pass a `model_config` to specify the form of the hypothesis passed to DeepSparse as part of the zero shot text classification scheme.

For instance, rather than running the comparison with `"This text is related to {}"`, we can instead use `"This text is similiar to {}"` with the following:

```python
from deepsparse import Pipeline, Context

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
pipeline = Pipeline.create(
  task="zero_shot_text_classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1,                # default batch size is 1
  model_config={"hypothesis_template": "This text is similar to {}"}
)

# run inference
prediction = pipeline(
    sequences="My favorite sports team is the Boston Red Sox",
    labels=["sports", "politics", "public health"]
)
print(prediction)

# sequences='My favorite sports team is the Boston Red Sox' labels=['sports', 'politics', 'public health'] scores=[0.5861895680427551, 0.32133620977401733, 0.0924743041396141]
```

### Cross Use Case Functionality
Check out the Pipeline User Guide for more details on configuring a Pipeline.

## DeepSparse Server
Built on the popular FastAPI and Uvicorn stack, DeepSparse Server enables you to set up a REST endpoint for serving inferences over HTTP. DeepSparse Server wraps the Pipeline API, so it inherits all the utilities provided by Pipelines.

The CLI command below launches a zero shot text classification pipeline with a 90% pruned-quantized oBERT model trained on MNLI:

```bash
deepsparse.server \
  --task zero_shot_text_classification \
  --model_path "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none" # or path/to/onnx
```

Making a request:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {
  "sequences": ["The Boston Red Sox are my favorite baseball team!"],
  "labels": ["sports", "politics", "public health"]
}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# {"sequences":["The Boston Red Sox are my favorite baseball team!"],"labels":[["sports","politics","public health"]],"scores":[[0.9649990200996399,0.028026442974805832,0.006974523887038231]]}
```

### Use Case Specific Arguments
To use the `labels` and `model_config` arguments in the server constructor, create a Server configuration file for passing the arguments via kwargs.

This configuration file sets the labels to `sports`, `politics` and `public health` and creates hypotheses of the form `"This sentence is similiar to {}"`.

```yaml
# config.yaml
endpoints:
  - task: zero_shot_text_classification
    model: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none
    kwargs:
      labels: ["sports", "politics", "public health"]
      model_config: {"hypothesis_template": "This text is similar to {}"}
```

Spin up the server:
```bash
deepsparse.server --config-file config.yaml
```

Making a request:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj =  {"sequences": ["The Boston Red Sox are my favorite baseball team!"]}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# {"sequences":["The Boston Red Sox are my favorite baseball team!"],"labels":[["sports","politics","public health"]],"scores":[[0.7818478941917419,0.17189143598079681,0.04626065865159035]]}

```
### Cross Use Case Functionality

Check out the Server User Guide for more details on configuring the Server.