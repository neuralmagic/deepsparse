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


<div style="display: flex; flex-direction: column; align-items: center;">
  <h1>
    <img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/icon-deepsparse.png" />
    &nbsp;&nbsp;DeepSparse
  </h1>
  <h3> An inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application</h3>
  <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap">
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

A CPU runtime that takes advantage of sparsity within neural networks to reduce compute. Read [more about sparsification](https://docs.neuralmagic.com/user-guide/sparsification).

Neural Magic's DeepSparse is able to integrate into popular deep learning libraries (e.g., Hugging Face, Ultralytics) allowing you to leverage DeepSparse for loading and deploying sparse models with ONNX. 
ONNX gives the flexibility to serve your model in a framework-agnostic environment. 
Support includes [PyTorch,](https://pytorch.org/docs/stable/onnx.html) [TensorFlow,](https://github.com/onnx/tensorflow-onnx) [Keras,](https://github.com/onnx/keras-onnx) and [many other frameworks](https://github.com/onnx/onnxmltools).

DeepSparse is available in two editions: 
1. [**DeepSparse Community**](#installation) is open-source and free for evaluation, research, and non-production use with our [DeepSparse Community License](https://neuralmagic.com/legal/engine-license-agreement/).
2. [**DeepSparse Enterprise**](https://docs.neuralmagic.com/products/deepsparse-ent) requires a Trial License or [can be fully licensed](https://neuralmagic.com/legal/master-software-license-and-service-agreement/) for production, commercial applications.

## Features

- 🔌 [DeepSparse Server](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server)
- 📜 [DeepSparse Benchmark](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)
- 👩‍💻 [NLP and Computer Vision Tasks Supported](https://github.com/neuralmagic/deepsparse/tree/main/examples)

## 🧰 Hardware Support and System Requirements

Review [Supported Hardware for DeepSparse](https://docs.neuralmagic.com/user-guide/deepsparse-engine/hardware-support) to understand system requirements. 
DeepSparse works natively on Linux; Mac and Windows require running Linux in a Docker or virtual machine; it will not run natively on those operating systems.

DeepSparse is tested on Python 3.7-3.10, ONNX 1.5.0-1.12.0, ONNX opset version 11+, and manylinux compliant. 
Using a [virtual environment](https://docs.python.org/3/library/venv.html) is highly recommended. 

## Installation

Install DeepSparse Community as follows: 

```bash
pip install deepsparse
```

To install the DeepSparse Enterprise, trial or inquire about licensing for DeepSparse Enterprise, see the [DeepSparse Enterprise documentation](https://docs.neuralmagic.com/products/deepsparse-ent).

## Features

### 🔌 DeepSparse Server

DeepSparse Server allows you to serve models and pipelines from the terminal. The server runs on top of the popular FastAPI web framework and Uvicorn web server. Install the server using the following command:

```bash
pip install deepsparse[server]
```

#### Single Model

Once installed, the following example CLI command is available for running inference with a single BERT model:

```bash
deepsparse.server \
    task question_answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
```

To look up arguments run: `deepsparse.server --help`.

#### Multiple Models
To serve multiple models in your deployment you can easily build a `config.yaml`. In the example below, we define two BERT models in our configuration for the question answering task:

```yaml
num_cores: 1
num_workers: 1
endpoints:
    - task: question_answering
      route: /predict/question_answering/base
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
    - task: question_answering
      route: /predict/question_answering/pruned_quant
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni
      batch_size: 1
```

Finally, after your `config.yaml` file is built, run the server with the config file path as an argument:
```bash
deepsparse.server config config.yaml
```

[Getting Started with DeepSparse Server](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server) for more info.

### 📜 DeepSparse Benchmark

The benchmark tool is available on your CLI to run expressive model benchmarks on DeepSparse with minimal parameters.

Run `deepsparse.benchmark -h` to look up arguments:

```shell
deepsparse.benchmark [-h] [-b BATCH_SIZE] [-shapes INPUT_SHAPES]
                          [-ncores NUM_CORES] [-s {async,sync}] [-t TIME]
                          [-nstreams NUM_STREAMS] [-pin {none,core,numa}]
                          [-q] [-x EXPORT_PATH]
                          model_path

```

[Getting Started with CLI Benchmarking](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark) includes examples of select inference scenarios: 
- Synchronous (Single-stream) Scenario
- Asynchronous (Multi-stream) Scenario


### 👩‍💻 NLP Inference Example

```python
from deepsparse import Pipeline

# SparseZoo model stub or path to ONNX file
model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

qa_pipeline = Pipeline.create(
    task="question-answering",
    model_path=model_path,
)

my_name = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```

NLP Tutorials:
- [Getting Started with Hugging Face Transformers 🤗](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers)

Tasks Supported: 
- [Token Classification: Named Entity Recognition](https://neuralmagic.com/use-cases/sparse-named-entity-recognition/)
- [Text Classification: Multi-Class](https://neuralmagic.com/use-cases/sparse-multi-class-text-classification/)
- [Text Classification: Binary](https://neuralmagic.com/use-cases/sparse-binary-text-classification/)
- [Text Classification: Sentiment Analysis](https://neuralmagic.com/use-cases/sparse-sentiment-analysis/)
- [Question Answering](https://neuralmagic.com/use-cases/sparse-question-answering/)

### 🦉 SparseZoo ONNX vs. Custom ONNX Models

DeepSparse can accept ONNX models from two sources: 

- **SparseZoo ONNX**: our open-source collection of sparse models available for download. [SparseZoo](https://github.com/neuralmagic/sparsezoo) hosts inference-optimized models, trained on repeatable sparsification recipes using state-of-the-art techniques from [SparseML](https://github.com/neuralmagic/sparseml).

- **Custom ONNX**: your own ONNX model, can be dense or sparse. Plug in your model to compare performance with other solutions.

```bash
> wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
Saving to: ‘mobilenetv2-7.onnx’
```

Custom ONNX Benchmark example:
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

The [GitHub repository](https://github.com/neuralmagic/deepsparse) includes package APIs along with examples to quickly get started benchmarking and inferencing sparse models.

### Scheduling Single-Stream, Multi-Stream, and Elastic Inference

DeepSparse offers up to three types of inferences based on your use case. Read more details here: [Inference Types](https://github.com/neuralmagic/deepsparse/blob/main/docs/source/scheduler.md).

1 ⚡ Single-stream scheduling: the latency/synchronous scenario, requests execute serially. [`default`]

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/single-stream.png" alt="single stream diagram" />

Use Case: It's highly optimized for minimum per-request latency, using all of the system's resources provided to it on every request it gets.

2 ⚡ Multi-stream scheduling: the throughput/asynchronous scenario, requests execute in parallel.

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/multi-stream.png" alt="multi stream diagram" />

PRO TIP: The most common use cases for the multi-stream scheduler are where parallelism is low with respect to core count, and where requests need to be made asynchronously without time to batch them.

3 ⚡ Elastic scheduling: requests execute in parallel, but not multiplexed on individual NUMA nodes.

Use Case: A workload that might benefit from the elastic scheduler is one in which multiple requests need to be handled simultaneously, but where performance is hindered when those requests have to share an L3 cache.

## Resources
#### Libraries
- [DeepSparse](https://docs.neuralmagic.com/deepsparse/)

- [SparseML](https://docs.neuralmagic.com/sparseml/)

- [SparseZoo](https://docs.neuralmagic.com/sparsezoo/)

- [Sparsify](https://docs.neuralmagic.com/sparsify/)


#### Versions
- [DeepSparse](https://pypi.org/project/deepsparse) | stable

- [DeepSparse-Nightly](https://pypi.org/project/deepsparse-nightly/) | nightly (dev)

- [GitHub](https://github.com/neuralmagic/deepsparse/releases) | releases

#### Info

- [Blog](https://www.neuralmagic.com/blog/) 

- [Resources](https://www.neuralmagic.com/resources/)


## Community

### Be Part of the Future... And the Future is Sparse!


Contribute with code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here.](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md)

For user help or questions about DeepSparse, sign up or log in to our **[Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)**. We are growing the community member by member and happy to see you there. Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/deepsparse/issues) You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, complete this [form.](http://neuralmagic.com/contact/)

### License

[DeepSparse Community](https://docs.neuralmagic.com/products/deepsparse) is licensed under the [Neural Magic DeepSparse Community License.](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC)
Some source code, example files, and scripts included in the deepsparse GitHub repository or directory are licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE) as noted.

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
