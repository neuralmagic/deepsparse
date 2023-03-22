# Deploying Transformers Embedding Extraction Models with DeepSparse

This page explains how to deploy a transformers embedding extraction Pipeline with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing
and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving
endpoint running DeepSparse with a single CLI.

## Installation Requirements
This use case requires the installation of [DeepSparse Server](/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](/user-guide/deepsparse-engine/hardware-support).

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
    model_path="zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
#     emb_extraction_layer=-1,         # (default: detect last layer)
#     extraction_strategy="per_token"  # (default: concat embedding for each token)
)

input_sequence = "The generalized embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 98304
```
An example returning the average embeddings of the tokens looks like this:
```python
from deepsparse import Pipeline

bert_emb_pipeline = Pipeline.create(
    task="transformers_embedding_extraction",
    model_path="zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
#     emb_extraction_layer=-1,         # (default: detect last layer)
    extraction_strategy="reduce_mean"
)

input_sequence = "The generalized embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 768
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
    model_path="zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
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

buckets = [16, 128]
bert_emb_pipeline = Pipeline.create(
    task="transformers_embedding_extraction",
    model_path="zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
#     emb_extraction_layer=-1,         # (default: detect last layer)
    extraction_strategy="reduce_mean",
    sequence_length = buckets
)

input_sequence = "The transformers embedding extraction Pipeline is the best!"
embedding = bert_emb_pipeline(input_sequence)
print(len(embedding.embeddings[0]))
# 768
```
### Cross Use Case Functionality
Check out the [Pipeline User Guide](/user-guide/deepsparse/deepsparse-pipelines) for more details on configuring a Pipeline.
## DeepSparse Server
DeepSparse Server is built on top of FastAPI and Uvicorn, enabling you to set up a REST endpoint for serving inferences over HTTP. 

The HTTP Server is a wrapper around the Pipeline. Therefore, it inherits all of the functionality we described above. It can be launched from the command line with a configuration file.

This configuration file sets sequence length to 64 and `emb_extraction_layer` to -3:
```yaml
# transformer-config.yaml
endpoints:
    - task: transformers_embedding_extraction
      model: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none
      kwargs:
        emb_extraction_layer: -3 # (default: detect last layer)
        sequence_length: 64       # uses sequence length 64
```

Spin up the server: 
```bash 
deepsparse.server --config_file transformer-config.yaml
```
Making requests: 

```python 
import requests, json
url = "http://0.0.0.0:5543/predict"
obj = {
  "inputs": "Mary is flying from Nairobi to New York to attend a conference on generative AI",
}
resp = requests.post(url=url, json=obj)
result = json.loads(resp.text)
print(len(result["embeddings"][0]))
# 49152
```
### Cross Use Case Functionality

Check out the [Server User Guide](/user-guide/deepsparse/deepsparse-server) for more details on configuring the Server.