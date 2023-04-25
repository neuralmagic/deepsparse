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

# Deploying Transformers Embedding Extraction Models with DeepSparse

This page explains how to deploy a transformers embedding extraction Pipeline with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing
and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving
endpoint running DeepSparse with a single CLI.

For the embedding extraction case, we will walk through an example of Pipeline and Server.

## Installation Requirements

This use case requires the installation of [DeepSparse Server](../../user-guide/installation.md).

Confirm your machine is compatible with our [hardware requirements](../../user-guide/hardware-support.md).

## DeepSparse Pipelines

Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of an embedding extraction Pipeline with a 80% pruned-quantized version of BERT trained on `wikipedia_bookcorpus`. We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input) is handled by the `Pipeline`.

With Transformers, you can use `task=transformer_embedding_extraction` for some extra utilities associated with embedding extraction.

The first utility is automatic embedding layer detection. If you set `emb_extraction_layer=-1` (the default), the Pipeline automatically detects the final transformer layer before the projection head and removes the projection head for you.

The second utility is automatic dimensionality reduction. You can use the `extraction_strategy` to perform a reduction on the sequence dimension rather than returning an embedding for each token. The options are:

- `per_token`: returns the embedding for each token in the sequence (default)
- `reduce_mean`: returns the average token of the sequence
- `reduce_max`: returns the max token of the sequence
- `cls_token`: returns the cls token from the sequence

An example using automatic embedding layer detection looks like this:

```python
from deepsparse import Pipeline

bert_emb_pipeline = Pipeline.create(
    task="transformers_embedding_extraction",
    model_path="zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
#     emb_extraction_layer=-1,         # (default: detect last layer)
#     extraction_strategy="per_token"  # (default: concat embedding for each token)
)

input_sequence = "The generalized embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 98304 << = 768*128 = hidden_dim * sequence_length>>
```

An example returning the average embeddings of the tokens looks like this:
```python
from deepsparse import Pipeline

bert_emb_pipeline = Pipeline.create(
    task="transformers_embedding_extraction",
    model_path="zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
#     emb_extraction_layer=-1,         # (default: detect last layer)
    extraction_strategy="reduce_mean"
)

input_sequence = "The generalized embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 768 <<=hidden dim>>
```

### Use Case Specific Arguments
The Transformers Embedding Extraction Pipeline contains additional arguments for configuring a `Pipeline`.

#### Sequence Length
The `sequence_length` argument adjusts the ONNX graph to handle a specific sequence length. In the DeepSparse Pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance.

The example below compiles the model and runs inference with sequence length of 64.
```python
from deepsparse import Pipeline

bert_emb_pipeline = Pipeline.create(
    task="transformers_embedding_extraction",
    model_path="zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
#     emb_extraction_layer=-1,         # (default: detect last layer)
    extraction_strategy="reduce_mean",
    sequence_length = 64
)

input_sequence = "The transformers embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 768
```

Alternatively, you can pass a list of sequence lengths, creating a "bucketable" pipeline. Under the hood, the DeepSparse Pipeline will compile multiple versions of the engine (utilizing a shared scheduler) and direct inputs towards the smallest bucket into which it fits.

The example below creates a bucket for smaller input lengths (16 tokens) and for larger input lengths (128 tokens).

```python
from deepsparse import Pipeline

bert_emb_pipeline = Pipeline.create(
    task="transformers_embedding_extraction",
    model_path="zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
#     emb_extraction_layer=-1,         # (default: detect last layer)
    extraction_strategy="reduce_mean",
    sequence_length = [16, 128]
)

input_sequence = "The transformers embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 768
```
### Cross Use Case Functionality
Check out the [Pipeline User Guide](../../user-guide/deepsparse-pipelines.md) for more details on configuring a Pipeline.

## DeepSparse Server
Built on the popular FastAPI and Uvicorn stack, DeepSparse Server enables you to set-up a REST endpoint  for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all of the utilities provided by Pipelines.

The CLI command below launches an embedding extraction pipeline with an 80% pruned-quantized BERT model identifed by its SparseZoo stub:

```bash
deepsparse.server \
  --task transformers_embedding_extraction \
  --model_path "zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni" # or path/to/onnx
```

You should see Uvicorn report that it is running on `http://0.0.0.0:5543`. Once launched, a `/docs` path is 
created with full endpoint descriptions and support for making sample requests.

Here is an example client request, using the Python `requests` library for formatting the HTTP:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {"inputs": "The transformers embedding extraction Pipeline is the best!"}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# >> {[[0.022315271198749542,0.02142658829689026, ... ,0.01950429379940033]]}
```

### Use Case Specific Arguments

To use the `sequence_length` and `extraction_strategy` arguments, we can a Server configuration file, passing the arguments via `kwargs`

This configuration file sets sequence length to 64 and returns all scores:

```yaml
# config.yaml
endpoints:
    - task: transformers_embedding_extraction
      model: zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni
      kwargs:
        sequence_length: 64       # uses sequence length 64
        extraction_strategy: reduce_mean
```

Spin up the server: 
```bash 
deepsparse.server --config_file config.yaml
```
Making requests: 

```python 
import requests, json
# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {"inputs": "The transformers embedding extraction Pipeline is the best!"}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# >> {[[0.022315271198749542,0.02142658829689026, ... ,0.01950429379940033]]}
resp = requests.post(url=url, json=obj)
result = json.loads(resp.text)
print(len(result["embeddings"][0]))
# 768
```

### Cross Use Case Functionality

Check out the [Server User Guide](../../user-guide/deepsparse-server.md) for more details on configuring the Server.

## Using a Custom ONNX File 
Apart from using models from the SparseZoo, DeepSparse allows you to deploy transformer embedding extraction pipelines with custom ONNX files. 

The first step is to obtain the ONNX model. You can obtain the file by converting your model to ONNX after training. 

Download the [DistilBERT - wikipedia_bookcorpus](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fdistilbert-none%2Fpytorch%2Fhuggingface%2Fwikipedia_bookcorpus%2Fpruned80_quant-none-vnni) 
ONNX model for demonstration: 

```bash 
sparsezoo.download zoo:nlp/masked_language_modeling/distilbert-none/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni --save-dir ./transformers_embedding_extraction
```

The `deployment` folder contains the following required files: 
- `config.json`
- `tokenizer.json`
- `model.onnx`

Use the folder as the model path to the transformer embedding extraction pipeline:
```python
from deepsparse import Pipeline

bert_emb_pipeline = Pipeline.create(
    task="transformers_embedding_extraction",
    model_path="transformers_embedding_extraction/deployment",
    emb_extraction_layer=-1,         # (default: detect last layer)
    extraction_strategy="per_token"  # (default: concat embedding for each token)
)

input_sequence = "The generalized embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 98304
```
