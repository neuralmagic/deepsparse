# Deploying Text Classification Models with DeepSparse

This page explains how to benchmark and deploy a text classification model with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API. It enables you to compile a model and run inference on raw input tensors.

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

As a baseline, let's check out ONNX Runtime's performance on oBERT. Make sure you have ORT installed (`pip install onnxruntime`).

```bash
deepsparse.benchmark \
  zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none \
  -b 64 -s sync -nstreams 1 \
  -e onnxruntime

> Original Model Path: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 19.3496
```

ONNX Runtime achieves 19 items/second with batch 64 and sequence length 128.

### DeepSparse Speedup
Now, let's run DeepSparse on an inference-optimized sparse version of oBERT. This model has been 90% pruned and quantized, while retaining >99% accuracy of the dense baseline on the MNLI dataset.
```bash
deepsparse.benchmark \
  zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none \
  -b 64 -s sync -nstreams 1 \
  -e deepsparse

> Original Model Path: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 124.0120
```
DeepSparse achieves 124 items/second, an 6.5x speed-up over ONNX Runtime!

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
compiled_model = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)

# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = compiled_model(inputs)
print(output)
# [array([[-0.9264987, -1.6990623,  2.3935342]], dtype=float32)]

```
## DeepSparse Pipelines
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of a text classification Pipeline with a 90% pruned-quantized version of oBERT. We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input) is handled by the `Pipeline`.

The Text Classification Pipeline can handle multi-input and single-input as well as single-label and multi-label classification.

#### Single-Input Single-Label Example (SST2)

Here's an example with a single input and single label prediction with a model trained on SST2:

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1                 # default batch size is 1
)

# run inference on image file
sequences = ["I think DeepSparse Pipelines are awesome!"]
prediction = pipeline(sequences)
print(prediction)
# labels=['0'] scores=[0.9986200332641602]

```

#### Multi-Input Single-Label Example (MNLI)

Here's an example with a single input and single label prediction with a model trained on MNLI:

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

# run inference
sequences = [[
  "The text classification pipeline is fast and easy to use!",
  "The pipeline for text classification makes it simple to get started"
]]
prediction = pipeline(sequences)
print(prediction)

# labels=['entailment'] scores=[0.6885718107223511]
```

#### Multi-Input Single-Label Example (QQP)

Here's an example with a single input and single label prediction with a model trained on QQP:

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

# run inference
sequences = [[
  "Which is the best gaming laptop under 40k?",
  "Which is the best gaming laptop under 40,000 rs?",
]]
prediction = pipeline(sequences)
print(prediction)

# labels=['duplicate'] scores=[0.9978139996528625]
```

### Single-Input Multi-Label Example (GoEmotions)

Here's an example with a single input and multi label prediction with a model trained on GoEmotions:

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
prediction = pipeline(["I am so glad you came today"])
print(prediction)

# labels=['joy'] scores=[0.9472543597221375]
```

### Use Case Specific Arguments
The Text Classification Pipeline contains additional arguments for configuring a `Pipeline`.

#### Sequence Length
The `sequence_length` argument adjusts the ONNX graph to handle a specific sequence length. In the DeepSparse Pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance. The defaul sequence length for text classification is 128.

The example below runs document classification using a model trained on IMBD at sequence length 512.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/pruned90_quant-none"
pipeline = Pipeline.create(
    task="text_classification",
    model_path=sparsezoo_stub,
    sequence_length=512,
    batch_size=1,
)

# run inference on image file
sequences = [[
  "I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered 'controversial' I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it's not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn't have much of a plot."
]]
prediction = pipeline(sequences)
print(prediction)
# labels=['0'] scores=[0.9984526634216309] (negative prediction)
```

Alternatively, you can pass a list of sequence lengths, creating a "bucketable" pipeline. Under the DeepSparse Pipeline will compile multiple versions of the model (utilizing a shared scheduler) and direct inputs towards the smallest bucket into which an input fits.

The example below creates a bucket for smaller input lengths (16 tokens) and for larger input lengths (128 tokens).
```python
from deepsparse import Pipeline, Context

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none"
pipeline = Pipeline.create(
  task="text-classification",
  model_path=sparsezoo_stub,
  batch_size=1,
  sequence_length=[32, 128],
  context=Context(num_streams=1)
)
# run inference on image file
prediction = pipeline([[
        "Timely access to information is in the best interests of the agencies",
        "It is everyone's best interest to get info in a timely manner",
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
  batch_size=2,                 # default batch size is 1
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
sequences_b2 = sequences_b1 * 2
prediction_b2 = pipeline_b2(sequences_b2)
print(prediction_b2)
# labels=[['entailment', 'neutral', 'contradiction']] scores=[[0.9688315987586975, 0.030656637623906136, 0.0005117706023156643]]
# labels=[['entailment', 'neutral', 'contradiction'], ['entailment', 'neutral', 'contradiction']] scores=[[0.9688315987586975, 0.030656637623906136, 0.0005117706023156643], [0.9688315987586975, 0.030656637623906136, 0.0005117706023156643]]
```

### Cross Use Case Functionality
Check out the Pipeline User Guide for more details on configuring a Pipeline.

## DeepSparse Server
Built on the popular FastAPI and Uvicorn stack, DeepSparse Server enables you to set up a REST endpoint for serving inferences over HTTP. DeepSparse Server wraps the Pipeline API, so it inherits all the utilities provided by Pipelines.

#### Single Input Usage

The CLI command below launches a single-input text classification pipeline with a 90% pruned-quantized oBERT model trained on SST2:

```bash
deepsparse.server \
  --task text-classification \
  --model_path "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none" # or path/to/onnx
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
# {"labels":["positive"],"scores":[0.9330279231071472]}
```

## Multi-Input Usage 

The CLI command below launches a single-input text classification pipeline with a 90% pruned-quantized oBERT model trained on MNLI:

```bash
deepsparse.server \
  --task text-classification \
  --model_path "zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none" # or path/to/onnx
```

Making a request:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {
  "sequences": [[
      "The text classification pipeline is fast and easy to use!",
      "The pipeline for text classification makes it simple to get started"
]]}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# {"labels":["entailment"],"scores":[0.6885718107223511]}
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
deepsparse.server --config-file config.yaml

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

Check out the Server User Guide for more details on configuring the Server.
