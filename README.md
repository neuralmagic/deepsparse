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

# ![icon for DeepSparse](https://github.com/neuralmagic/deepsparse/blob/main/docs/icon-engine.png) DeepSparse Engine

### CPU inference engine that delivers unprecedented performance for sparse models

<br>
<p>
    <a href="https://github.com/neuralmagic/deepsparse/blob/master/LICENSE-NEURALMAGIC"><img alt="GitHub" src="https://img.shields.io/static/v1.svg?label=LICENSE&message=neural%20magic%20engine&color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/comingsoon.svg?color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://docs.neuralmagic.com/deepsparse/">
        <img alt="Documentation" src="https://img.shields.io/website/http/neuralmagic.com/engine/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/engine.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic.com/deepsparse/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
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

## Overview

The DeepSparse Engine is a CPU runtime that delivers unprecedented performance by taking advantage of natural sparsity within neural networks to reduce compute required as well as accelerate memory bound workloads. It is focused on model deployment and scaling machine learning pipelines, fitting seamlessly into your existing deployments as an inference backend.

This repository includes package APIs along with examples to quickly get started learning about and actually running sparse models.

### Related Products

- [SparseZoo](https://github.com/neuralmagic/sparsezoo):
  Neural network model repository for highly sparse models and optimization recipes
- [SparseML](https://github.com/neuralmagic/sparseml):
  Libraries for state-of-the-art deep neural network optimization algorithms,
  enabling simple pipelines integration with a few lines of code
- [Sparsify](https://github.com/neuralmagic/sparsify):
  Easy-to-use autoML interface to optimize deep neural networks for
  better inference performance and a smaller footprint

## Compatibility

The DeepSparse Engine ingests models in the [ONNX](https://onnx.ai/) format, allowing for compatibility with [PyTorch](https://pytorch.org/docs/stable/onnx.html), [TensorFlow](https://github.com/onnx/tensorflow-onnx), [Keras](https://github.com/onnx/keras-onnx), and [many other frameworks](https://github.com/onnx/onnxmltools) that support it. This reduces the extra work of preparing your trained model for inference to just one step of exporting.

## Quick Tour

To expedite inference and benchmarking on real models, we include the `sparsezoo` package. [SparseZoo](https://github.com/neuralmagic/sparsezoo) hosts inference optimized models, trained on repeatable optimization recipes using state-of-the-art techniques from [SparseML](https://github.com/neuralmagic/sparseml).

### Quickstart with SparseZoo ONNX Models

**MobileNetV1 Dense**

Here is how to quickly perform inference with DeepSparse Engine on a pre-trained dense MobileNetV1 from SparseZoo.

```python
from deepsparse import compile_model
from sparsezoo.models import classification
batch_size = 64

# Download model and compile as optimized executable for your machine
model = classification.mobilenet_v1()
engine = compile_model(model, batch_size=batch_size)

# Fetch sample input and predict output using engine
inputs = model.data_inputs.sample_batch(batch_size=batch_size)
outputs, inference_time = engine.timed_run(inputs)
```

**MobileNetV1 Optimized**

When exploring available optimized models, you can use the `Zoo.search_optimized_models` utility to find models that share a base. 

Let us try this on the dense MobileNetV1 to see what is available.

```python
from sparsezoo import Zoo
from sparsezoo.models import classification
print(Zoo.search_optimized_models(classification.mobilenet_v1()))
```
Output:
```
[Model(stub=cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none),
 Model(stub=cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-conservative),
 Model(stub=cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate),
 Model(stub=cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate)]
```

Great. We can see there are two pruned versions targeting FP32, `conservative` at 100% and `moderate` at >= 99% of baseline accuracy. There is also a `pruned_quant` variant targetting INT8.

Let's say you want to evaluate best performance on FP32 and are okay with a small drop in accuracy, so we can choose `pruned-moderate` over `pruned-conservative`.

```python
from deepsparse import compile_model
from sparsezoo.models import classification
batch_size = 64

model = classification.mobilenet_v1(optim_name="pruned", optim_category="moderate")
engine = compile_model(model, batch_size=batch_size)

inputs = model.data_inputs.sample_batch(batch_size=batch_size)
outputs, inference_time = engine.timed_run(inputs)
```

### Quickstart with custom ONNX models

We accept ONNX files for custom models, too. Simply plug in your model to compare performance with other solutions.

```bash
> wget https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
Saving to: ‘mobilenetv2-7.onnx’
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

For a more in-depth read on available APIs and workflows, check out the [examples](examples/) and [DeepSparse Engine documentation](https://docs.neuralmagic.com/deepsparse/deepsparse).

## Hardware Support

The DeepSparse Engine is validated to work on x86 Intel and AMD CPUs running Linux operating systems.

It is highly recommended to run on a CPU with AVX-512 instructions available for optimal algorithms to be enabled. 

Here is a table detailing specific support for some algorithms over different microarchitectures:

|   x86 Extension    |          Microarchitectures         | Activation Sparsity | Kernel Sparsity | Sparse Quantization |
|:------------------:|:-----------------------------------:|:-------------------:|:---------------:|:-------------------:|
|      [AMD AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)      |             [Zen 2](https://en.wikipedia.org/wiki/Zen_2), [Zen 3](https://en.wikipedia.org/wiki/Zen_3)            |    not supported    |    optimized    |    not supported    |
|     [Intel AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)     |          [Haswell](https://en.wikipedia.org/wiki/Haswell_(microarchitecture)), [Broadwell](https://en.wikipedia.org/wiki/Broadwell_(microarchitecture)), and newer         |    not supported    |    optimized    |    not supported    |
|    [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512)   |         [Skylake](https://en.wikipedia.org/wiki/Skylake_(microarchitecture)), [Cannon Lake](https://en.wikipedia.org/wiki/Cannon_Lake_(microarchitecture)), and newer        |      optimized      |    optimized    |       emulated      |
| [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) VNNI (DL Boost) | [Cascade Lake](https://en.wikipedia.org/wiki/Cascade_Lake_(microarchitecture)), [Ice Lake](https://en.wikipedia.org/wiki/Ice_Lake_(microprocessor)), [Cooper Lake](https://en.wikipedia.org/wiki/Cooper_Lake_(microarchitecture)), [Tiger Lake](https://en.wikipedia.org/wiki/Tiger_Lake_(microprocessor)) |      optimized      |    optimized    |      optimized      |

## Installation

This repository is tested on Python 3.6+, and ONNX 1.5.0+. It is recommended to install in a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your system in order.

Install with pip using:

```bash
pip install deepsparse
```

Then if you want to explore the [examples](examples/), clone the repository and any install additional dependencies found in example folders.

## Notebooks

For some step-by-step examples, we have Jupyter [notebooks](notebooks/) showing how to compile models with the DeepSparse Engine, check the predictions for accuracy, and benchmark them on your hardware.
  
## Available Models and Recipes

A number of pre-trained baseline and recalibrated models models in the [SparseZoo](https://github.com/neuralmagic/sparsezoo) can be used with the engine for higher performance. The types available for each model architecture are noted in its [SparseZoo model repository listing](https://github.com/neuralmagic/sparsezoo/blob/main/docs/available-models.md).

## Resources and Learning More

- DeepSparse Engine [Documentation](https://docs.neuralmagic.com/deepsparse/deepsparse), [Notebooks](notebooks/), [Examples](examples/)
- APIs to Invoke and Run the DeepSparse Engine
- Passing a Model to the Deep Sparse Engine and Deploying
- Debugging and Optimizing Performance
- [SparseML Documentation](https://docs.neuralmagic.com/sparseml/sparseml)
- [Sparsify Documentation](https://docs.neuralmagic.com/sparsify/sparsify)
- [SparseZoo Documentation](https://docs.neuralmagic.com/sparsezoo/sparsezoo)
- Neural Magic [Blog](https://www.neuralmagic.com/blog/), [Resources](https://www.neuralmagic.com/resources/), [Website](https://www.neuralmagic.com/)

## Contributing

We appreciate contributions to the code, examples, and documentation as well as bug reports and feature requests! [Learn how here](CONTRIBUTING.md).

## Join the Community

For user help or questions about the DeepSparse Engine, use our [GitHub Discussions](https://www.github.com/neuralmagic/deepsparse/discussions/). Everyone is welcome!

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please email us at [learnmore@neuralmagic.com](mailto:learnmore@neuralmagic.com) or fill out this [form](http://neuralmagic.com/contact/).

## License

The project's binary containing the DeepSparse Engine is licensed under the [Neural Magic Engine License 1.0](LICENSE-NEURALMAGIC). 

Example files and scripts included in this repository are licensed under the [Apache License Version 2.0](LICENSE) as noted.

## Release History

Official builds are hosted on PyPi
- stable: [deepsparse](https://pypi.org/project/deepsparse/)
- nightly (dev): [deepsparse-nightly](https://pypi.org/project/deepsparse-nightly/)

[Track this project via GitHub Releases.](https://github.com/neuralmagic/deepsparse/releases/)

## Citation

Find this project useful in your research or other communications? Please consider citing Neural Magic's [paper](http://proceedings.mlr.press/v119/kurtz20a.html):

```bibtex
@inproceedings{pmlr-v119-kurtz20a, 
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
    pdf = {http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf},, 
    url = {http://proceedings.mlr.press/v119/kurtz20a.html}, 
    abstract = {Optimizing convolutional neural networks for fast inference has recently become an extremely active area of research. One of the go-to solutions in this context is weight pruning, which aims to reduce computational and memory footprint by removing large subsets of the connections in a neural network. Surprisingly, much less attention has been given to exploiting sparsity in the activation maps, which tend to be naturally sparse in many settings thanks to the structure of rectified linear (ReLU) activation functions. In this paper, we present an in-depth analysis of methods for maximizing the sparsity of the activations in a trained neural network, and show that, when coupled with an efficient sparse-input convolution algorithm, we can leverage this sparsity for significant performance gains. To induce highly sparse activation maps without accuracy loss, we introduce a new regularization technique, coupled with a new threshold-based sparsification method based on a parameterized activation function called Forced-Activation-Threshold Rectified Linear Unit (FATReLU). We examine the impact of our methods on popular image classification models, showing that most architectures can adapt to significantly sparser activation maps without any accuracy loss. Our second contribution is showing that these these compression gains can be translated into inference speedups: we provide a new algorithm to enable fast convolution operations over networks with sparse activations, and show that it can enable significant speedups for end-to-end inference on a range of popular models on the large-scale ImageNet image classification task on modern Intel CPUs, with little or no retraining cost.} 
}
```
