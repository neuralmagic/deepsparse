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


<div align="center">
    <h1><img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/icon-deepsparse.png" />&nbsp;&nbsp;DeepSparse Engine</h1>
	<p>
		<b>
        <h3> Sparsity-aware neural network inference engine for GPU-class performance on CPUs </h3>
        </b>
	</p>

<p>
    <a href="https://docs.neuralmagic.com/deepsparse/">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ/">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/issues/">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height=25>
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

<br>
</div>

A CPU runtime that takes advantage of sparsity within neural networks to reduce compute. Read more about sparsification [here](https://docs.neuralmagic.com/main/source/getstarted.html#sparsification).

Neural Magic's DeepSparse Engine is able to integrate into popular deep learning libraries (e.g., Hugging Face, Ultralytics) allowing you to leverage DeepSparse for loading and deploying sparse models with ONNX. ONNX gives the flexibility to serve your model in a framework-agnostic environment. Support includes [PyTorch,](https://pytorch.org/docs/stable/onnx.html) [TensorFlow,](https://github.com/onnx/tensorflow-onnx) [Keras,](https://github.com/onnx/keras-onnx) and [many other frameworks](https://github.com/onnx/onnxmltools).

## Features

- 🔌 [DeepSparse Server](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server)
- 📜 [DeepSparse Benchmark](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark_model)
- 👩‍💻 [NLP and Computer Vision Tasks Supported](https://github.com/neuralmagic/deepsparse/tree/main/examples)
- 🧰 [CPU Hardware Support for Various Architectures](https://docs.neuralmagic.com/deepsparse/source/hardware.html)

## Installation
The DeepSparse Engine is tested on Python 3.6-3.9, ONNX 1.5.0-1.10.1, and manylinux compliant. Using a [virtual environment](https://docs.python.org/3/library/venv.html) is highly recommended. Install the engine using the following command:

```bash
pip install deepsparse
```

## 🔌 DeepSparse Server

The DeepSparse Server allows you to serve models and pipelines from the terminal. The server runs on top of the popular FastAPI web framework and Uvicorn web server. Install the server using the following command:

```bash
pip install deepsparse[server]
```

### Single Model

Once installed, the following example CLI command is available for running inference with a single BERT model:

```bash
deepsparse.server \
    --task question_answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
```

To look up arguments run: `deepsparse.server --help`.

### Multiple Models
To serve multiple models in your deployment you can easily build a `config.yaml`. In the example below, we define two BERT models in our configuration for the question answering task:

```yaml
models:
    - task: question_answering
      model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      alias: question_answering/base
    - task: question_answering
      model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni
      batch_size: 1
      alias: question_answering/pruned_quant
```

Finally, after your `config.yaml` file is built, run the server with the config file path as an argument:
```bash
deepsparse.server --config_file config.yaml
```

[Getting Started with the DeepSparse Server](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server) for more info.

## 📜 DeepSparse Benchmark

The benchmark tool is available on your CLI to run expressive model benchmarks on the DeepSparse Engine with minimal parameters.

Run `deepsparse.benchmark -h` to look up arguments:

```shell
deepsparse.benchmark [-h] [-b BATCH_SIZE] [-shapes INPUT_SHAPES]
                          [-ncores NUM_CORES] [-s {async,sync}] [-t TIME]
                          [-nstreams NUM_STREAMS] [-pin {none,core,numa}]
                          [-q] [-x EXPORT_PATH]
                          model_path

```

[Getting Started with CLI Benchmarking](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark_model) includes examples of select inference scenarios: 
- Synchronous (Single-stream) Scenario
- Asynchronous (Multi-stream) Scenario


## 👩‍💻 NLP Inference Example

```python
from deepsparse.transformers import pipeline

# SparseZoo model stub or path to ONNX file
model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

qa_pipeline = pipeline(
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

## 🦉 SparseZoo ONNX vs. Custom ONNX Models

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
Compatibility/Support Notes:
- ONNX version 1.5-1.7
- ONNX opset version 11+
- ONNX IR version has not been tested at this time

The [GitHub repository](https://github.com/neuralmagic/deepsparse) includes package APIs along with examples to quickly get started benchmarking and inferencing sparse models.

## Scheduling Single-Stream, Multi-Stream, and Elastic Inference

The DeepSparse Engine offers up to three types of inferences based on your use case. Read more details here: [Inference Types](https://github.com/neuralmagic/deepsparse/blob/main/docs/source/scheduler.md).

1 ⚡ Single-stream scheduling: the latency/synchronous scenario, requests execute serially. [`default`]

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/single-stream.png" alt="single stream diagram" />

Use Case: It's highly optimized for minimum per-request latency, using all of the system's resources provided to it on every request it gets.

2 ⚡ Multi-stream scheduling: the throughput/asynchronous scenario, requests execute in parallel.

<img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/multi-stream.png" alt="multi stream diagram" />

PRO TIP: The most common use cases for the multi-stream scheduler are where parallelism is low with respect to core count, and where requests need to be made asynchronously without time to batch them.

3 ⚡ Elastic scheduling: requests execute in parallel, but not multiplexed on individual NUMA nodes.

Use Case: A workload that might benefit from the elastic scheduler is one in which multiple requests need to be handled simultaneously, but where performance is hindered when those requests have to share an L3 cache.

## 🧰 CPU Hardware Support

With support for AVX2, AVX-512, and VNNI instruction sets, the DeepSparse Engine is validated to work on x86 Intel (Haswell generation and later) and AMD CPUs running Linux. Mac and Windows require running Linux in a Docker or virtual machine.

Here is a table detailing specific support for some algorithms over different microarchitectures:

|   x86 Extension    |          Microarchitectures         | Activation Sparsity | Kernel Sparsity | Sparse Quantization |
|:------------------:|:-----------------------------------:|:-------------------:|:---------------:|:-------------------:|
|      [AMD AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)      |             [Zen 2,](https://en.wikipedia.org/wiki/Zen_2) [Zen 3](https://en.wikipedia.org/wiki/Zen_3)            |    not supported    |    optimized    |    emulated    |
|     [Intel AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)     |          [Haswell,](<https://en.wikipedia.org/wiki/Haswell_(microarchitecture)>) [Broadwell,](<https://en.wikipedia.org/wiki/Broadwell_(microarchitecture)>) and newer         |    not supported    |    optimized    |    emulated    |
|    [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512)   |         [Skylake,](<https://en.wikipedia.org/wiki/Skylake_(microarchitecture)>) [Cannon Lake,](<https://en.wikipedia.org/wiki/Cannon_Lake_(microarchitecture)>) and newer        |      optimized      |    optimized    |       emulated      |
| [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) VNNI (DL Boost) | [Cascade Lake,](<https://en.wikipedia.org/wiki/Cascade_Lake_(microarchitecture)>) [Ice Lake,](<https://en.wikipedia.org/wiki/Ice_Lake_(microprocessor)>) [Cooper Lake,](<https://en.wikipedia.org/wiki/Cooper_Lake_(microarchitecture)>) [Tiger Lake](<https://en.wikipedia.org/wiki/Tiger_Lake_(microprocessor)>) |      optimized      |    optimized    |      optimized      |

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

The project's binary containing the DeepSparse Engine is licensed under the [Neural Magic Engine License.](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC) Example files and scripts included in this repository are licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE) as noted.
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
