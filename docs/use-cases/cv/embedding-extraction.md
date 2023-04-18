# Deploying Embedding Extraction Models With DeepSparse
This page explains how to deploy an Embedding Extraction Pipeline with DeepSparse.

## Installation Requirements
This use case requires the installation of [DeepSparse Server](/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](/user-guide/deepsparse-engine/hardware-support).

## Model Format
The Embedding Extraction Pipeline enables you to generate embeddings in any domain, meaning you can use it with any ONNX model. It (optionally) removes the projection head from the model, such that you can re-use SparseZoo models and custom models you have trained in the embedding extraction scenario.

There are two options for passing a model to the Embedding Extraction Pipeline:

- Pass a Local ONNX File
- Pass a SparseZoo Stub (which identifies an ONNX model in the SparseZoo)

## DeepSparse Pipelines
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of an embedding extraction Pipeline with a 95% pruned-quantized version of ResNet-50 trained on `imagenet`. We can then pass images the `Pipeline` and receive the embeddings. All of the pre-processing is handled by the `Pipeline`.

The Embedding Extraction Pipeline handles some useful actions around inference:

- First, on initialization, the Pipeline (optionally) removes a projection head from a model. You can use the `emb_extraction_layer` argument to specify which layer to return. If your ONNX model has no projection head, you can set `emb_extraction_layer=None` (the default) to skip this step.

- Second, as with all DeepSparse Pipelines, it handles pre-processing such that you can pass raw input. You will notice that in addition to the typical task argument used in `Pipeline.create()`, the Embedding Extraction Pipeline includes a `base_task` argument. This argument tells the Pipeline the domain of the model, such that the Pipeline can figure out what pre-processing to do.

This is an example of extracting the last layer from ResNet-50:

Download an image to use with the Pipeline.
```bash
wget https://huggingface.co/spaces/neuralmagic/image-classification/resolve/main/lion.jpeg
```

Run the following to extract the embedding: 
```python
from deepsparse import Pipeline

# this step removes the projection head before compiling the model
rn50_embedding_pipeline = Pipeline.create(
    task="embedding-extraction",
    base_task="image-classification", # tells the pipeline to expect images and normalize input with ImageNet means/stds
    model_path="zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none",
    emb_extraction_layer=-3, # extracts last layer before projection head and softmax
)

# this step runs pre-processing, inference and returns an embedding
embedding = rn50_embedding_pipeline(images="lion.jpeg")
print(len(embedding.embeddings[0][0]))
# 2048 << size of final layer>>
```

### Cross Use Case Functionality
Check out the [Pipeline User Guide](../../user-guide/deepsparse-pipelines.md) for more details on configuring the Pipeline.

## DeepSparse Server
As an alternative to the Python API, DeepSparse Server allows you to serve an Embedding Extraction Pipeline over HTTP. Configuring the server uses the same parameters and schemas as the Pipelines. 

Once launched, a `/docs` endpoint is created with full endpoint descriptions and support for making sample requests.

This configuration file sets `emb_extraction_layer` to -3:
```yaml
# config.yaml
endpoints:
    - task: embedding_extraction
      model: zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none
      kwargs:
        base_task: image_classification
        emb_extraction_layer: -3
```
Spin up the server: 
```bash 
deepsparse.server --config_file config.yaml
```

Make requests to the server: 
```python
import requests, json
url = "http://0.0.0.0:5543/predict/from_files"
paths = ["lion.jpeg"]
files = [("request", open(img, 'rb')) for img in paths]
resp = requests.post(url=url, files=files)
result = json.loads(resp.text)

print(len(result["embeddings"][0][0]))

# 2048 << size of final layer>>
```

### Cross Use Case Functionality
Check out the [Server User Guide](../../user-guide/deepsparse-server.md) for more details on configuring the Server.
