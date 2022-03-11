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


<h1><img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/icon-deepsparse.png" />&nbsp;&nbsp;DeepSparse</h1>

<p>
    <a href="https://docs.neuralmagic.com/deepsparse/">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ/">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://discuss.neuralmagic.com/">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=discourse" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/actions/workflows/quality-check.yaml">
        <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/deepsparse/Quality%20Checks/main?label=build&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/deepsparse.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
    </a>
    <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
        <img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=25>
    </a>
     <a href="https://medium.com/limitlessai">
        <img src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height=25>
    </a>
    <a href="https://twitter.com/neuralmagic">
        <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25>
    </a>
</p>

<h3> DeepSparse is a deep learning inference engine for running super-fast sparse models. 🚀🚀🚀</h3>

A CPU runtime engine delivering GPU-class performance by taking advantage of sparsity within neural networks to reduce compute required as well as accelerate memory bound workloads. Read more about sparsification [here](https://docs.neuralmagic.com/main/source/getstarted.html#sparsification).

## Install 💽
Tested on Python 3.6+, and ONNX 1.5.0+. [virtual environment](https://docs.python.org/3/library/venv.html) is highly recommended!!

```bash
pip install deepsparse
```
__ __
## Quick Start 💻

One of the greatest features of our inference engine is the ability to integrate into popular deep learning libraries (e.g. Hugging Face, Ultralytics) allowing you to leverage DeepSparse for loading and deploying sparse models with ONNX runtime (ORT). ORT gives flexibility to serve your model in a framework agnostic environment. Support includes [PyTorch,](https://pytorch.org/docs/stable/onnx.html) [TensorFlow,](https://github.com/onnx/tensorflow-onnx) [Keras,](https://github.com/onnx/keras-onnx) and [many other frameworks](https://github.com/onnx/onnxmltools).

To begin your inference adventure with DeepSparse is super simple. You can either run DeepSparse in CLI or  in Python. 

Let's first check out the CLI features... 👇
__ __

## Working in CLI with DeepSparse Server & Benchmark

After you've installed DeepSparse in your Python environment, you have two resources via CLI at your disposal:
- `deepsparse.server` :
    - If you are interested in serving your models. 
- `deepsparse.benchmark` :
    - If you are interested in experimenting with the throughput/latency performance of your models under different inference scenarios.

### 1. DeepSparse Server

The DeepSparse inference server allows you to serve models and pipelines in deployment. The server runs on top of the popular FastAPI web framework and Uvicorn web server.

 - run `deepsparse.server -h` to lookup arguments.

##### ⭐ Single Model ⭐

Example CLI command for running inference with a single model:

```bash
deepsparse.server --task question_answering --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"
```

##### ⭐ Multiple Models ⭐
To serve multiple models in your deployment you can easily build a `config.yaml`. 
In the example below, we are defining 2 BERT models in our configuration:

    models:
    - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
        batch_size: 1
        alias: question_answering/dense
    - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95
        batch_size: 1
        alias: question_answering/sparse_quantized

Finally, after your `config.yaml` file is built, you can run the server with the config file path as an argument:
```bash
deepsparse.server --config_file config.yaml
```

### 2. Deepsparse Benchmark

The benchmark tool is available on your CLI to run expressive model benchmarks on the DeepSparse Engine with minimal parameters.

Run `deepsparse.benchmark -h` to look up arguments.


```shell
deepsparse.benchmark [-h] [-b BATCH_SIZE] [-shapes INPUT_SHAPES]
                          [-ncores NUM_CORES] [-s {async,sync}] [-t TIME]
                          [-nstreams NUM_STREAMS] [-pin {none,core,numa}]
                          [-q] [-x EXPORT_PATH]
                            model_path

```

- [Getting Started with CLI Benchmarking](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark_model) Includes examples of inference scenarios such as: 
    - Synchronous (Single-stream) Scenario
    - Asynchronous (Multi-stream) Scenario

Let's now try running DeepSparse via Python... 👇
__ __

## NLP Inference 👩‍💻 Question Answering

```python
from deepsparse.transformers import pipeline

# SparseZoo model stub or path to ONNX file
onnx_filepath="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98"

qa_pipeline = pipeline(
    task="question-answering",
    model_path=onnx_filepath,
    num_cores=None, # uses all available CPU cores by default
)

my_name = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```

NLP Tutorials:
- [Getting Started with Hugging Face Transformers 🤗](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers)

Tasks Supported: 
- Text Classification (Sentiment Analysis)
- Question Answering
- Masked Language Modeling (MLM)
__ __
## CV Inference 👩‍💻 Object Detection (Placeholder)

```python
from deepsparse.transformers import pipeline

# SparseZoo model stub or path to ONNX file
onnx_filepath="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98"

qa_pipeline = pipeline(
    task="question-answering",
    model_path=onnx_filepath,
    num_cores=None, # uses all available CPU cores by default
)

my_name = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```
CV Tutorials:
- [Getting Started with MobileNet v2](https://github.com/neuralmagic/deepsparse/tree/main/examples/classification)
- [Getting Started with YOLOv3](https://github.com/neuralmagic/deepsparse/tree/main/examples/detection)
- [Getting Started with YOLOv3 <LINK DOESN'T WORK>](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolov3)

Tasks Supported: 
- Image Classification
- Object Detection
- Image Segmentation
__ __

## SparseZoo ONNX vs. Custom ONNX Models

DeepSparse can accept ONNX models from two sources: 

1. `SparseZoo ONNX`: our open-source collection of sparse models available for download. [SparseZoo](https://github.com/neuralmagic/sparsezoo) hosts inference-optimized models, trained on repeatable sparsification recipes using state-of-the-art techniques from [SparseML.](https://github.com/neuralmagic/sparseml)

2. `Custom ONNX`: Your own onnx model, can be dense or sparse. Plug in your model to compare performance with other solutions. 👇


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
Compatibility/Support Notes
- ONNX version 1.5-1.7
- ONNX opset version 11+
- ONNX IR version has not been tested at this time

For a more in-depth read on available APIs and workflows, check out the [examples](https://github.com/neuralmagic/deepsparse/blob/main/examples/) and [DeepSparse Engine documentation.](https://docs.neuralmagic.com/deepsparse)



__ __

## Scheduling Single-Stream, Multi-Stream, and Elastic Inference ⚡

The DeepSparse Engine offers up to 3 types of inferences based on your use-case. You can read more details here: [Inference Types](https://github.com/neuralmagic/deepsparse/blob/main/docs/source/scheduler.md).

1 ⚡ Single stream scheduling: the latency/synchronous scenario, requests execute serially. [`default`]

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/single-stream.png" alt="single stream diagram" />

PRO TIP: It's highly optimized for minimum per-request latency, using all of the system's resources provided to it on every request it gets.

2 ⚡ Multi-stream scheduling: the throughput/asynchronous scenario, requests execute in parallel.

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/multi-stream.png" alt="multi stream diagram" />

PRO TIP: The most common use cases for the multi-stream scheduler are where parallelism is low with respect to core count, and where requests need to be made asynchronously without time to batch them.

3 ⚡ Elastic scheduling: requests execute in parallel, but not multiplexed on individual NUMA Nodes.

PRO TIP: A workload that might benefit from the elastic scheduler is one in which multiple requests need to be handled simultaneously, but where performance is hindered when those requests have to share an L3 cache.
__ __

## Hardware Support 🧰

The DeepSparse Engine is validated to work on x86 Intel and AMD CPUs running Linux operating systems. Mac and Windows require running Linux in a Docker or virtual machine.

It is highly recommended to run on a CPU with AVX-512 instructions available for optimal algorithms to be enabled. 

Here is a table detailing specific support for some algorithms over different microarchitectures:

|   x86 Extension    |          Microarchitectures         | Activation Sparsity | Kernel Sparsity | Sparse Quantization |
|:------------------:|:-----------------------------------:|:-------------------:|:---------------:|:-------------------:|
|      [AMD AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)      |             [Zen 2,](https://en.wikipedia.org/wiki/Zen_2) [Zen 3](https://en.wikipedia.org/wiki/Zen_3)            |    not supported    |    optimized    |    not supported    |
|     [Intel AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)     |          [Haswell,](<https://en.wikipedia.org/wiki/Haswell_(microarchitecture)>) [Broadwell,](<https://en.wikipedia.org/wiki/Broadwell_(microarchitecture)>) and newer         |    not supported    |    optimized    |    not supported    |
|    [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512)   |         [Skylake,](<https://en.wikipedia.org/wiki/Skylake_(microarchitecture)>) [Cannon Lake,](<https://en.wikipedia.org/wiki/Cannon_Lake_(microarchitecture)>) and newer        |      optimized      |    optimized    |       emulated      |
| [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) VNNI (DL Boost) | [Cascade Lake,](<https://en.wikipedia.org/wiki/Cascade_Lake_(microarchitecture)>) [Ice Lake,](<https://en.wikipedia.org/wiki/Ice_Lake_(microprocessor)>) [Cooper Lake,](<https://en.wikipedia.org/wiki/Cooper_Lake_(microarchitecture)>) [Tiger Lake](<https://en.wikipedia.org/wiki/Tiger_Lake_(microprocessor)>) |      optimized      |    optimized    |      optimized      |

## Resources 🔧

<table>
<tr><th> Documentation </th><th> &emsp;&emsp;&emsp;Versions </th><th> Info </th></tr>
<tr><td>

[DeepSparse](https://docs.neuralmagic.com/deepsparse/)

[SparseML](https://docs.neuralmagic.com/sparseml/)

[SparseZoo](https://docs.neuralmagic.com/sparsezoo/)

[Sparsify](https://docs.neuralmagic.com/sparsify/)

</td><td>

&emsp;stable : : [DeepSparse](https://pypi.org/project/deepsparse)

&emsp;nightly (dev) : : [DeepSparse-Nightly](https://pypi.org/project/deepsparse-nightly/)

&emsp;releases : : [GitHub](https://github.com/neuralmagic/deepsparse/releases)

</td><td>

[Blog](https://www.neuralmagic.com/blog/) 
[Resources](https://www.neuralmagic.com/resources/)

</td></tr> </table>



## Community 🧙

### Be Part of the Future... And the Future is Sparse!


Contribute with code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here.](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md)

### Join

For user help or questions about DeepSparse, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). We are growing the community member by member and happy to see you there. Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/deepsparse/issues)

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please fill out this [form.](http://neuralmagic.com/contact/)

### License

The project's binary containing the DeepSparse Engine is licensed under the [Neural Magic Engine License.](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC)

Example files and scripts included in this repository are licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE) as noted.
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
    url = {http://proceedings.mlr.press/v119/kurtz20a.html},
}
```