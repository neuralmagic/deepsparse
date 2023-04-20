# How to use the scheduler across engine, pipeline, server
In DeepSparse the scheduler determines the Engine's execution strategy. For most synchronous cases, the default `single_stream` is recommended.
For running a model server or parallel inferences, try `multi_stream` for
maximum utilization of hardware.

The available options are: 

- default: maps to `single_stream`
- `single_stream`: requests from separate threads execute serially
- `multi_stream`: requests from separate threads execute in parallel
- `elastic`: requests from separate threads are distributed across NUMA nodes

Here are examples of how to use the `scheduler` argument with DeepSparse. 
## Engine 
Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended using the Pipeline API but Engine is available if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 90% pruned-quantized oBERT trained on `sst2` from SparseZoo with the `scheduler` set to `multi_stream`:
```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
batch_size = 1
bert_engine = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size,   # defaults to batch size 1,
  scheduler="multi_stream" # default: maps to single_stream
)

# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = bert_engine(inputs)
print(output)
# [array([[-0.34614536,  0.09025408]], dtype=float32)]
```
## Pipeline 
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input) is handled by the `Pipeline`.

We will use the `Pipeline.create()` constructor to create an instance of a sentiment analysis Pipeline with a 90% pruned-quantized version of oBERT trained on `sst2` with the scheduler set to `single_stream`. 

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
batch_size = 1
sa_pipeline = Pipeline.create(
  task="sentiment-analysis",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=1,                # default batch size is 1
  scheduler="single_stream" # default: maps to single_stream
)

# run inference on image file
prediction = sa_pipeline("The sentiment analysis pipeline is fast and easy to use")
print(prediction)
# labels=['positive'] scores=[0.9955807328224182]
```
## Server
DeepSparse Server is built on top of FastAPI and Uvicorn, enabling you to set up a REST endpoint for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all the utilities provided by Pipelines.


The first step is to define a configuration file that defines the desired `scheduler`: 
```yaml
# sentiment-analysis-config.yaml
endpoints:
  - task: sentiment-analysis
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
    scheduler: multi_stream # default: maps to single_stream
```
The CLI command below launches a sentiment analysis pipeline with a 90% pruned-quantized oBERT model with the configuration file.
```bash
deepsparse.server \
  --config-file sentiment-analysis-config.yaml
```
You should see Uvicorn report that it is running on http://0.0.0.0:5543. Once launched, a /docs path is created with full endpoint descriptions and support for making sample requests.

Run inference: 
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
