# Deploying Embedding Extraction Models with DeepSparse

This page explains how to create and deploy an Embedding Extraction Pipeline with DeepSparse. 

## Installation Requirements

This section requires the [DeepSparse Server Install](/get-started/install/deepsparse).

## Getting Started

Before you start using DeepSparse, confirm your machine is
compatible with our [hardware requirements](/user-guide/deepsparse-engine/hardware-support).

### Model Format

The Embedding Extraction Pipeline is a general Pipeline, meaning you can pass an ONNX model
for any domain. The Embedding Extraction Pipeline (optionally) removes the projection head from the model, 
making it easy to re-use SparseZoo models and models you have trained for Embedding Extraction.

The model can be provided in two formats:
- Pass a Local ONNX File (see task Use Cases pages for usage in each domain)
- Pass a SparseZoo Stub (which identifies the ONNX file on the SparseZoo)

The examples below use option 2.

## Deployment APIs

DeepSparse provides both a Python `Pipeline` API and an out-of-the-box
HTTP Server. Both options provide similar specifications for configurations.

### Python API

`Pipelines` are the default interface for running inference with DeepSparse.

Once a model is obtained, either through SparseML training or directly from SparseZoo, 
Pipelines can be used to handle pre-processing and post-processing of input, making it easy to add DeepSparse to your application.
In addition to typical Pipeline functionality, the Embeddeding Extraction Pipeline can also optionally remove the 
projection head from the model, enabling you to re-use a SparseZoo model or trained model with the Embedding
Extraction Pipeline.

### HTTP Server

As an alternative to the Python API, DeepSparse Server allows you to
serve an Embedding Extraction Pipeline over HTTP. Configuring the server uses the same parameters and schemas as the `Pipelines`. 
Once launched, a `/docs` endpoint is created with full endpoint descriptions and support for making sample requests.

For full documentation on deploying sparse image classification models with the
DeepSparse Server, see the [documentation for DeepSparse Server](/user-guide/deploying-deepsparse/deepsparse-server).

## Deployment Examples

The following section includes example usage of the `Pipeline` and Server APIs for various use cases. 
Each example uses a SparseZoo stub to pull down the model,
but a local path to an ONNX file can also be passed as the `model_path`.

### Python API

The Embedding Extraction Pipeline handles some useful actions. First, on load, the `Pipeline` 
(optionally) removes a projection head from a model. Then, during inference, the `Pipeline` handles the pre-processing 
(e.g., with ResNet subtracting by ImageNet means, dividing by ImageNet standard deviation) and (optionally) can perform a reduction of the embedding
(e.g., with BERT averaging over token embeddings or returning the `cls` token).

You will notice that in addition to the typical `task` argument used in `Pipeline.create()`, the Embedding Extraction Pipeline includes a 
`base_task` argument. This argument tells the Pipeline the domain of the model, such that the Pipeline 
can figure out what pre-processing to do.

An example Image Embedding Extraction Pipeline with ResNet-50:

```python
from deepsparse import Pipeline

# creates pipeline, chopping off the projection head
rn50_embedding_pipeline = Pipeline.create(
    task='embedding_extraction',
    base_task='image_classification', # Base Task that model is typically used for
    model_path='model_path='zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none',  # Path to checkpoint or SparseZoo stub
    embed_extraction_layer=-1 # extract last layer
    # extraction_strategy='no_reduction` # defaults to no reduction
)

input_image = "my_image.png" # path to input image
embedding = rn50_embedding_pipeline(images=input_image) # returns final layer of resnet-model
```

An example Text Embedding Extraction Pipeline with BERT:

```python
from deepsparse import Pipeline
bert_embedding_pipeline = Pipeline.create(
    task='embedding_extraction',
    base_task='nlp', # Base task that model is typically used for
    model_path='model_path='zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni',  # Path to checkpoint or SparseZoo stub
    embed_extraction_layer=-1 # extract last layer 
    extraction_strategy=`reduce_mean` # options are no_reduction (return vector of token embeddings), reduce_mean (avg of token embeddings), reduce_cls (cls embedding) 
)

input_sequence = "The generalized embedding extraction Pipeline is the best!"
embedding = bert_embedding_pipeline(images=input_sequence) # returns the average of the token embeddings from BERT
```

Note: if you have an ONNX model with no projection head already, pass **`XXX TO BE UPDATED XXX`**.

Note: supported `base_task` include:
- `image_classification`
- `yolo`
- `nlp`
- `custom`
- **`XXX TO BE UPDATED XXX`**

### HTTP Server

The HTTP Server is a wrapper around the Pipeline.

Spinning up:
```bash
deepsparse.server \
    task embedding_extraction \
    --model_path "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none" \
    --base_task image_classification
    --embed_extraction_layer -1
```

Making a request:
```python
import requests

url = 'http://0.0.0.0:5543/predict/from_files'
path = ['goldfish.jpeg'] # just put the name of images in here
files = [('request', open(img, 'rb')) for img in path]
resp = requests.post(url=url, files=files)
```

## Benchmarking

The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs.
Want to find out how fast our sparse ONNX models perform inference? You can quickly run benchmarking tests on your own with a single CLI command.

You only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:
```bash
deepsparse.benchmark zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni
```

To learn more about benchmarking, refer to the appropriate documentation.
Also, check out our [Benchmarking Tutorial on GitHub  ](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)!
