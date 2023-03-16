# DeepSparse User Guide

This directory demonstrates usage of DeepSparse's key API, including:
- [Benchmarking CLI](#performance-benchmarking)
- [Engine API](#engine)
- [Pipeline API](#pipeline)
- [Server API](#server)

## Installation

Install via `pip`. Using a virtual enviornment is highly recommended.

```bash
pip install deepsparse
```

See the [DeepSparse Installation page](https://docs.neuralmagic.com/get-started/install/deepsparse) for further installation options.

## [Performance Benchmarking](deepsparse-benchmarking.md)

DeepSparse's key feature is its performance on commodity CPUs. DeepSparse is competitive with other CPU runtimes
like ONNX Runtime for unoptimized dense models. However, when optimization techniques like pruning and quantization 
are applied to a model, DeepSparse can achieve an order-of-magnitude speedup. As an example, let's compare DeepSparse and ORT's performance on BERT. In SparseZoo, there is [90% pruned-quantized BERT](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fpruned90_quant-none). 

Running this model on an AWS `c6i.16xlarge` instance, DeepSparse achieves a ***12x speedup*** over ORT!

ORT achieves 18.5 items/second running BERT (make sure you have ORT installed `pip install onnxruntime`):
```bash
deepsparse.benchmark zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none -b 64 -s sync -nstreams 1 -i [64,384] -e onnxruntime

>> Original Model Path: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 18.5742
```

DeepSparse achieves 226 items/second running the pruned-quantized version of BERT:

```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none -b 64 -s sync -nstreams 1 -e onnxruntime

>> Original Model Path: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/pruned90_quant-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 226.6340
```

**Pro-Tip:** In place of a [SparseZoo](https://sparsezoo.neuralmagic.com/) stubs, you can pass a local ONNX file to test your model.

## Deployment APIs

Now that we have seen DeepSparse's performance gains, let's take a look at how we can add DeepSparse to an application.

DeepSparse includes three deployment APIs:
- Engine is the lowest-level API. With Engine, you pass tensors and recieve the raw logits.
- Pipeline wraps the Engine with pre- and post-processing. With Pipeline, you pass raw data and
recieve the prediction.
- Server wraps Pipelines with a REST API using FastAPI. With Server, you send raw data over HTTP
and recieve the prediction.

Let's walk through a simple example of each API to give a sense of usage. As an example, we will use
the sentiment analysis use-case with a 90% pruned-quantized version of BERT. 

### Engine

Engine is the lowest-level API, allowing you to run inference directly on input tensors.

The example below downloads a 90% pruned-quantized BERT model for sentiment analysis 
in ONNX format from SparseZoo, compiles the model, and runs inference on randomly generated input.

```python
from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs, model_to_path

# download onnx, compile model
zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  
batch_size = 1
bert_model = compile_model(
  model=zoo_stub,         # sparsezoo stub or path/to/local/model.onnx
  batch_size=batch_size   # default is batch 1
)

# run inference (input is raw numpy tensors, output is raw scores)
inputs = generate_random_inputs(model_to_path(zoo_stub), batch_size)
output = bert_model(inputs)
print(output)

# > [array([[-0.3380675 ,  0.09602544]], dtype=float32)] << raw scores
```

#### Model Format

DeepSparse can accept ONNX models from two sources:

- **SparseZoo Stubs**: SparseZoo is Neural Magic's open-source repository of sparse models. You can pass a SparseZoo stub, a unique identifier for
each model to DeepSparse, which downloads the necessary ONNX files from the remote repository. 

- **Custom ONNX**: DeepSparse allows you to use your own model in ONNX format. Checkout the SparseML user guide for more details on exporting
your sparse models to ONNX format. Here's a quick example using a custom ONNX file from the ONNX model zoo:

```bash
wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
> Saving to: ‘mobilenetv2-7.onnx’
```

```python
from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs
onnx_filepath = "mobilenetv2-7.onnx"
batch_size = 16

# Generate random sample input
inputs = generate_random_inputs(onnx_filepath, batch_size)

# Compile and run
engine = compile_model(onnx_filepath, batch_size)
outputs = engine.run(inputs)
```

### [Pipeline](deepsparse-pipeline.md)

Pipeline is the default API for interacting with DeepSparse. Similar to Hugging Face Pipelines,
DeepSparse Pipelines wrap Engine with pre- and post-processing (as well as other utilities), 
enabling you to send raw data to DeepSparse and recieve the post-processed prediction.

The example below downloads a 90% pruned-quantized BERT model for sentiment analysis 
in ONNX format from SparseZoo, sets up a pipeline, and runs inference on sample data.

```python
from deepsparse import Pipeline

# download onnx, set up pipeline
zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  
batch_size = 1
sentiment_analysis_pipeline = Pipeline.create(
  task="sentiment-analysis",    # name of the task
  model_path=zoo_stub,          # zoo stub or path to local onnx file
  batch_size=batch_size         # default is batch 1
)

# run inference (input is a sentence, output is the prediction)
prediction = sentiment_analysis_pipeline("I love using DeepSparse Pipelines")
print(prediction)
# > labels=['positive'] scores=[0.9954759478569031]
```


### [Server](deepsparse-server.md)

Server wraps Pipelines with REST APIs, that make it easy to stand up a model serving endpoint
running DeepSparse. This enables you to send raw data to DeepSparse over HTTP and recieve the post-processed
predictions.

DeepSparse Server is launched from the command line, configured via arguments or a server configuration file.

The following downloads a 90% pruned-quantized BERT model for sentiment analysis in ONNX format
from SparseZoo and launches a sentiment analysis endpoint:

```bash
deepsparse.server \
  --task sentiment-analysis \
  --model_path zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

Alternatively, the following configuration file can launch the Server.

```yaml
# config.yaml
endpoints:
  - task: sentiment-analysis
    route: /predict
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

Spinning up:
```bash
deepsparse.server \
  --config-file config.yaml
```

You should see Uvicorn report that it is running on port 5543. Navigating to the `/docs` endpoint will
show the exposed routes as well as sample requests.

We can then send a request over HTTP. In this example, we will use the Python requests package
to format the HTTP.

```python
import requests

url = "http://localhost:5543/predict" # Server's port default to 5543
obj = {"sequences": "Snorlax loves my Tesla!"}

response = requests.post(url, json=obj)
print(response.text)
# {"labels":["positive"],"scores":[0.9965094327926636]}
```

## Supported Tasks

DeepSparse supports many common CV and NLP use cases out of the box. Check out the use case guide for more details on the task-specific APIs.

**Computer Vision**:
- Image Classification: `task="image_classification"`
- Object Detection: `task="yolo"`
- Instance Segmentation: `task="yolact"`

**Natural Language Processing**:
- Embedding Extraction: `task="transformers_embedding_extraction"`
- Text Classification: `task="text-classification"`
- Zero Shot Text Classification: `task="zero-shot-text-classification"` 
- Sentiment Analysis: `task="sentiment-analysis"`
- Token Classification: `task="token-classification"`
- Question Answering: `task="question-answering"`
