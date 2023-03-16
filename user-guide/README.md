---
title: "DeepSparse Community"
metaTitle: "DeepSparse Community"
metaDescription: "Sparsity-aware neural network inference engine for GPU-class performance on CPUs"
index: 1000
---

<div style="display: flex; flex-direction: column;">
  <h1>
    <img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/icon-deepsparse.png" />
    &nbsp;&nbsp;DeepSparse Community
  </h1>
  <h3>A deep learning inference runtime with GPU-class performance on CPUs</h3>
  <div style="display: flex; flex-wrap: wrap">
    <a href="https://docs.neuralmagic.com/deepsparse/">
      <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height="25" />
    </a>
    <a href="https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ/">
      <img alt="Slack" src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height="25" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/issues/">
      <img alt="Support" src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height="25" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/actions/workflows/quality-check.yaml">
      <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/deepsparse/Quality%20Checks/main?label=build&style=for-the-badge" height="25" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/releases">
      <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/deepsparse.svg?style=for-the-badge" height="25" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/blob/main/CODE_OF_CONDUCT.md">
      <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height="25" />
    </a>
    <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
      <img alt="YouTube" src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height="25" />
    </a>
    <a href="https://medium.com/limitlessai">
      <img alt="Medium" src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height="25" />
    </a>
    <a href="https://twitter.com/neuralmagic">
      <img alt="Twitter" src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height="25" />
    </a>
  </div>
</div>

**DeepSparse is a CPU deep learning inference runtime with GPU-class performance.**

Because DeepSparse reaches GPU-class latency with commodity CPUs, you no longer need to tie deployments to accelerators to reach the performance needed for production.
Free from specialized hardware, deployments can then take advantage of the flexibility and scalability of software-defined inference:
- Deploy the same model and runtime on any hardware from Intel to AMD to ARM (soon) and from cloud to data center to edge, including on pre-existing systems
- Scale vertically to 192 cores, horizontally with Kubernetes, or abstractly with serverless
- Integrate easily into "deploy with code" provisioning systems
- No wrestling with drivers, operator support, and compatibility issues

Simply put, with DeepSparse on CPUs, you can both simplify your deep learning deployment process and save on infrastructure costs required to support enterprise-scale workloads.

## Hardware Support and System Requirements

Review [CPU Hardware Support for Various Architectures](https://docs.neuralmagic.com/deepsparse/source/hardware.html) to understand system requirements.

DeepSparse runs natively on Linux. Mac and Windows require running Linux in a Docker or virtual machine.

DeepSparse is tested on Python 3.7-3.10, ONNX 1.5.0-1.12.0, ONNX opset version 11+, and manylinux compliant systems. Using a virtual environment is highly recommended.

## Installation

Install DeepSparse Community with `pip`:

```bash
pip install deepsparse
```

See the [DeepSparse Installation page](https://docs.neuralmagic.com/get-started/install/deepsparse) for further installation options.

## Performance Benchmarking

DeepSparse's key feature is its performance on commodity CPUs. DeepSparse is competitive with other CPU runtimes
like ONNX Runtime for unoptimized dense models. However, when optimization techniques like pruning and quantization 
are applied to a model, DeepSparse can achieve an order-of-magnitude speedup.

As an example, let's compare DeepSparse and ORT's performance on BERT. In SparseZoo, there is 90% pruned and quantized 
BERT which retains >99% of the accuracy of the baseline dense model. 
Running this model on a `c6i.16xlarge` instance, DeepSparse achieves a ***12x speedup*** over ORT!

![Performance Benchmarking Chart](https://raw.githubusercontent.com/neuralmagic/docs/rs/use-case-update/src/images/bert-performance-chart.png)

To replicate the results, make sure you ONNX Runtime installed (`pip install onnxruntime`).

ORT achieves 18.5 items/second running BERT:
```bash
deepsparse.benchmark zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none -b 64 -s sync -nstreams 1 -i [64,384] -e onnxruntime

>> Original Model Path: zoo:nlp/text_classification/obert-base/pytorch/huggingface/mnli/base-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 18.5742
```

DeepSparse achieves 226 items/second running the pruned-quantized version of BERT:

```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none -b 32 -s sync -nstreams 1 -e onnxruntime

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
- Server wraps Pipelines with a REST API using FastAPI. With Server, you send raw data to an endpoint over HTTP
and recieve the prediction.

Let's walk through a simple example of each API to give a sense of usage. As an example, we will use
the sentiment analysis use-case with a 90% pruned-quantized version of BERT. 

Check out the [use case section](/use-cases) for details on the APIs of each supported use case.

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

### Pipeline

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


### Server

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

DeepSparse supports many common CV and NLP use cases out of the box. Follow the links below for 
usage examples of each use case.

**Computer Vision**:
- [Image Classification](/use-cases/image-classification/deploying): `task="image_classification"`
- [Object Detection](/use-cases/object-detection/deploying): `task="yolo"`
- [Instance Segmentation](/use-cases/instance-segmentation/deploying): `task="yolact"`

**Natural Language Processing**:
- [Embedding Extraction](/use-cases/embedding-extraction): `task="transformers_embedding_extraction"`
- [Text Classification](/use-cases/use-cases/natural-language-processing/text-classification): `task="text-classification"`
- [Zero Shot Text Classification](/use-cases/use-cases/natural-language-processing/zero-shot-text-classification): `task="zero-shot-text-classification"` 
- [Sentiment Analysis](/use-cases/use-cases/natural-language-processing/sentiment-analysis): `task="sentiment-analysis"`
- [Token Classification](/use-cases/use-cases/natural-language-processing/token-classification): `task="token-classification"`
- [Question Answering](/use-cases/use-cases/natural-language-processing/question-answering): `task="question-answering"`

## Advanced Functionality

DeepSparse contains many additional utilties that simplify deployment. Check out the user guide for detailed exploration of the feature set:
- DeepSparse Pipelines
- DeepSparse Server
- DeepSparse Benchmark
- Logging
- Scheduler

## Training a Sparse Model

For details on training a sparse model, check out the SparseML user guide.

## Community

### Be Part of the Future ... And the Future is Sparse!

Contribute with code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here.](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md)

For user help or questions about DeepSparse, sign up or log into our [Neural Magic Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). We are growing the community member by member and happy to see you there. Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/deepsparse/issues) You can get the latest news, webinar and event invites, research papers, and other ML performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, complete this [form.](http://neuralmagic.com/contact/)

### License

DeepSparse Community is licensed under the [Neural Magic DeepSparse Community License.](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC)
Some source code, example files, and scripts included in the DeepSparse GitHub repository or directory are licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE) as noted.

[DeepSparse Enterprise](https://docs.neuralmagic.com/products/deepsparse-ent) requires a Trial License or [can be fully licensed](https://neuralmagic.com/legal/master-software-license-and-service-agreement/) for production, commercial applications.

### Cite

Find this project useful in your research or other communications? Please consider citing:

```bibtex
@InProceedings{
    pmlr-v119-kurtz20a,
    title = {Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks},
    author = {Kurtz, Mark and Kopinsky, Justin and Gelashvili, Rati and Matveev, Alexander and Carr, John and Goin, Michael and Leiserson, William and Moore, Sage and Nell, Bill and Shavit, Nir and Alistarh, Dan},
    booktitle = {Proceedings of the 37th International Conference on Machine Learning},
    pages = {5533--5543},
    year = {2020},
    editor = {Hal Daumé III and Aarti Singh},
    volume = {119},
    series = {Proceedings of Machine Learning Research},
    address = {Virtual},
    month = {13--18 Jul},
    publisher = {PMLR},
    pdf = {http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf},
    url = {http://proceedings.mlr.press/v119/kurtz20a.html}
}

@article{DBLP:journals/corr/abs-2111-13445,
  author    = {Eugenia Iofinova and
               Alexandra Peste and
               Mark Kurtz and
               Dan Alistarh},
  title     = {How Well Do Sparse Imagenet Models Transfer?},
  journal   = {CoRR},
  volume    = {abs/2111.13445},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.13445},
  eprinttype = {arXiv},
  eprint    = {2111.13445},
  timestamp = {Wed, 01 Dec 2021 15:16:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-13445.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
