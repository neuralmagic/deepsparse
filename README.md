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
  <h4> An inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application</h4>
  <div align="center">
    <a href="https://docs.neuralmagic.com/deepsparse/">
      <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height="20" />
    </a>
    <a href="https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ/">
      <img alt="Slack" src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height="20" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/issues/">
      <img alt="Support" src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height="20" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/actions/workflows/quality-check.yaml">
      <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/deepsparse/Quality%20Checks/main?label=build&style=for-the-badge" height="20" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/releases">
      <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/deepsparse.svg?style=for-the-badge" height="20" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/blob/main/CODE_OF_CONDUCT.md">
      <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height="20" />
    </a>
    <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
      <img alt="YouTube" src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height="20" />
    </a>
    <a href="https://medium.com/limitlessai">
      <img alt="Medium" src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height="20" />
    </a>
    <a href="https://twitter.com/neuralmagic">
      <img alt="Twitter" src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height="20" />
    </a>
  </div>
</div>

A CPU runtime that takes advantage of sparsity within neural networks to reduce compute. Read [more about sparsification](https://docs.neuralmagic.com/user-guides/sparsification).

Neural Magic's DeepSparse is able to integrate into popular deep learning libraries (e.g., Hugging Face, Ultralytics) allowing you to leverage DeepSparse for loading and deploying sparse models with ONNX. 
ONNX gives the flexibility to serve your model in a framework-agnostic environment. 
Support includes [PyTorch,](https://pytorch.org/docs/stable/onnx.html) [TensorFlow,](https://github.com/onnx/tensorflow-onnx) [Keras,](https://github.com/onnx/keras-onnx) and [many other frameworks](https://github.com/onnx/onnxmltools).

## Installation

Install DeepSparse Community as follows: 

```bash
pip install deepsparse
```

DeepSparse is available in two editions: 
1. [**DeepSparse Community**](#installation) is open-source and free for evaluation, research, and non-production use with our [DeepSparse Community License](https://neuralmagic.com/legal/engine-license-agreement/).
2. [**DeepSparse Enterprise**](https://docs.neuralmagic.com/products/deepsparse-ent) requires a Trial License or [can be fully licensed](https://neuralmagic.com/legal/master-software-license-and-service-agreement/) for production, commercial applications.

## 🧰 Hardware Support and System Requirements

To ensure that your CPU is compatible with DeepSparse, it is recommended to review the [Supported Hardware for DeepSparse](https://docs.neuralmagic.com/user-guide/deepsparse-engine/hardware-support) documentation.

To ensure that you get the best performance from DeepSparse, it has been thoroughly tested on Python versions 3.7-3.10, ONNX versions 1.5.0-1.12.0, ONNX opset version 11 or higher, and manylinux compliant systems. It is highly recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html) when running DeepSparse. Please note that DeepSparse is only supported natively on Linux. For those using Mac or Windows, running Linux in a Docker or virtual machine is necessary to use DeepSparse.

## Features

- 👩‍💻 Pipelines for [NLP](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/transformers), [CV Classification](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/image_classification), [CV Detection](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/yolo), [CV Segmentation](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/yolact) and more!
- 🔌 [DeepSparse Server](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server)
- 📜 [DeepSparse Benchmark](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)
- ☁️ [Cloud Deployments and Demos](https://github.com/neuralmagic/deepsparse/tree/main/examples)

### 👩‍💻 Pipelines

Pipelines are a high-level Python interface for running inference with DeepSparse across select tasks in NLP and CV:

|          NLP          |            CV             |
|-----------------------|---------------------------|
|  Text Classification `"text_classification"`  | Image Classification `"image_classification"`      |
|  Token Classification `"token_classification"` | Object Detection `"yolo"`          |
|  Sentiment Analysis `"sentiment_analysis"`   | Instance Segmentation `"yolact"`       |
|  Question Answering `"question_answering"`   | Keypoint Detection `"open_pif_paf"`       |
| MultiLabel Text Classification `"text_classification"` |                  |
| Document Classification `"text_classification"` |                         |
| Zero-Shot Text Classification `"zero_shot_text_classification"` |                         |


**NLP Example** | Question Answering
```python
from deepsparse import Pipeline

qa_pipeline = Pipeline.create(
    task="question-answering",
    model_path="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni",
)

inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```
**CV Example** | Image Classification

```python
from deepsparse import Pipeline

cv_pipeline = Pipeline.create(
  task='image_classification', 
  model_path='zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none',
)

input_image = "my_image.png"
inference = cv_pipeline(images=input_image)
```


### 🔌 DeepSparse Server

DeepSparse Server is a tool that enables you to serve your models and pipelines directly from your terminal.

The server is built on top of two powerful libraries: the FastAPI web framework and the Uvicorn web server. This combination ensures that DeepSparse Server delivers excellent performance and reliability. Install with this command:

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
To deploy multiple models in your setup, a `config.yaml` file should be created. In the example provided, two BERT models are configured for the question-answering task:

```yaml
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

After the `config.yaml` file has been created, the server can be started by passing the file path as an argument:
```bash
deepsparse.server config config.yaml
```

Read the [DeepSparse Server](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server) README for further details.

### 📜 DeepSparse Benchmark

DeepSparse Benchmark, a command-line (CLI) tool, is used to evaluate the DeepSparse Engine's performance with ONNX models. This tool processes arguments, downloads and compiles the network into the engine, creates input tensors, and runs the model based on the selected scenario. 

Run `deepsparse.benchmark -h` to look up arguments:

```shell
deepsparse.benchmark [-h] [-b BATCH_SIZE] [-i INPUT_SHAPES] [-ncores NUM_CORES] [-s {async,sync,elastic}] [-t TIME]
                     [-w WARMUP_TIME] [-nstreams NUM_STREAMS] [-pin {none,core,numa}] [-e ENGINE] [-q] [-x EXPORT_PATH]
                     model_path

```


Refer to the [Benchmark](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark) README for examples of specific inference scenarios.

### 🦉 Custom ONNX Model Support

DeepSparse is capable of accepting ONNX models from two sources:

**SparseZoo ONNX**: This is an open-source repository of sparse models available for download. [SparseZoo](https://github.com/neuralmagic/sparsezoo) offers inference-optimized models, which are trained using repeatable sparsification recipes and state-of-the-art techniques from [SparseML](https://github.com/neuralmagic/sparseml).

**Custom ONNX**: Users can provide their own ONNX models, whether dense or sparse. By plugging in a custom model, users can compare its performance with other solutions.

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

The [GitHub repository](https://github.com/neuralmagic/deepsparse) repository contains package APIs and examples that help users swiftly begin benchmarking and performing inference on sparse models.

### Scheduling Single-Stream, Multi-Stream, and Elastic Inference

DeepSparse offers different inference scenarios based on your use case. Read more details here: [Inference Types](https://github.com/neuralmagic/deepsparse/blob/main/docs/source/scheduler.md).

⚡ **Single-stream** scheduling: the latency/synchronous scenario, requests execute serially. [`default`]

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/single-stream.png" alt="single stream diagram" />

It's highly optimized for minimum per-request latency, using all of the system's resources provided to it on every request it gets.

⚡ **Multi-stream** scheduling: the throughput/asynchronous scenario, requests execute in parallel.

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/multi-stream.png" alt="multi stream diagram" />

The most common use cases for the multi-stream scheduler are where parallelism is low with respect to core count, and where requests need to be made asynchronously without time to batch them.

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
