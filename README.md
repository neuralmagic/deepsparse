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

# ![icon for DeepSparse](https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/source/icon-deepsparse.png) DeepSparse Engine

### Neural network inference engine that delivers unprecedented performance for sparsified models on CPUs

<p>
    <a href="https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC"><img alt="GitHub" src="https://img.shields.io/static/v1.svg?label=LICENSE&message=neural%20magic%20engine&color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/blob/main/LICENSE">
         <img alt="GitHub" src="https://img.shields.io/static/v1.svg?label=LICENSE&message=apache-2.0&color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://docs.neuralmagic.com/deepsparse/">
        <img alt="Documentation" src="https://img.shields.io/website/http/docs.neuralmagic.com/deepsparse/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/deepsparse.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/blob/main/CODE_OF_CONDUCT.md">
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

## Sparsification

Sparsification is the process of taking a trained deep learning model and removing redundant information from the overprecise and over-parameterized network resulting in a faster and smaller model.
Techniques for sparsification are all encompassing including everything from inducing sparsity using [pruning](https://neuralmagic.com/blog/pruning-overview/) and [quantization](https://arxiv.org/abs/1609.07061) to enabling naturally occurring sparsity using [activation sparsity](http://proceedings.mlr.press/v119/kurtz20a.html) or [winograd/FFT](https://arxiv.org/abs/1509.09308). 
When implemented correctly, these techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics.
For example, pruning plus quantization can give over [7x improvements in performance](resnet50link) while recovering to nearly the same baseline accuracy.

The Deep Sparse product suite builds on top of sparsification enabling you to easily apply the techniques to your datasets and models using recipe-driven approaches.
Recipes encode the directions for how to sparsify a model into a simple, easily editable format.
- Download a sparsification recipe and sparsified model from the [SparseZoo](https://github.com/neuralmagic/sparsezoo).
- Alternatively, create a recipe for your model using [Sparsify](https://github.com/neuralmagic/sparsify).
- Apply your recipe with only a few lines of code using [SparseML](https://github.com/neuralmagic/sparseml).
- Finally, for GPU-level performance on CPUs, deploy your sparse-quantized model with the [DeepSparse Engine](https://github.com/neuralmagic/deepsparse).


**Full Deep Sparse product flow:**  

<img src="https://docs.neuralmagic.com/docs/source/sparsification/flow-overview.svg" width="960px">

## Compatibility

The DeepSparse Engine ingests models in the [ONNX](https://onnx.ai/) format, allowing for compatibility with [PyTorch](https://pytorch.org/docs/stable/onnx.html), [TensorFlow](https://github.com/onnx/tensorflow-onnx), [Keras](https://github.com/onnx/keras-onnx), and [many other frameworks](https://github.com/onnx/onnxmltools) that support it. This reduces the extra work of preparing your trained model for inference to just one step of exporting.

## Quick Tour

To expedite inference and benchmarking on real models, we include the `sparsezoo` package. [SparseZoo](https://github.com/neuralmagic/sparsezoo) hosts inference-optimized models, trained on repeatable sparsification recipes using state-of-the-art techniques from [SparseML](https://github.com/neuralmagic/sparseml).

### Quickstart with SparseZoo ONNX Models

**ResNet-50 Dense**

Here is how to quickly perform inference with DeepSparse Engine on a pre-trained dense ResNet-50 from SparseZoo.

```python
from deepsparse import compile_model
from sparsezoo.models import classification

batch_size = 64

# Download model and compile as optimized executable for your machine
model = classification.resnet_50()
engine = compile_model(model, batch_size=batch_size)

# Fetch sample input and predict output using engine
inputs = model.data_inputs.sample_batch(batch_size=batch_size)
outputs, inference_time = engine.timed_run(inputs)
```

**ResNet-50 Sparsified**

When exploring available optimized models, you can use the `Zoo.search_optimized_models` utility to find models that share a base.

Let us try this on the dense ResNet-50 to see what is available.

```python
from sparsezoo import Zoo
from sparsezoo.models import classification

model = classification.resnet_50()
print(Zoo.search_optimized_models(model))
```

Output:

```shell
[
    Model(stub=cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none), 
    Model(stub=cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-conservative), 
    Model(stub=cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate), 
    Model(stub=cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate), 
    Model(stub=cv/classification/resnet_v1-50/pytorch/sparseml/imagenet-augmented/pruned_quant-aggressive)
]
```

We can see there are two pruned versions targeting FP32 and two pruned, quantized versions targeting INT8.
The `conservative`, `moderate`, and `aggressive` tags recover to 100%, >=99%, and <99% of baseline accuracy respectively.

For a version of ResNet-50 that recovers close to the baseline and is very performant, choose the pruned_quant-moderate model.
This model will run [nearly 7x faster](linktoresnet50example) than the baseline model on a compatible CPU (with the VNNI instruction set enabled).
For hardware compatibility, see the Hardware Support section.

```python
from deepsparse import compile_model
import numpy

batch_size = 64
sample_inputs = [numpy.random.randn(batch_size, 3, 224, 224).astype(numpy.float32)]

# run baseline benchmarking
engine_base = compile_model(
    model="zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none", 
    batch_size=batch_size,
)
benchmarks_base = engine_base.benchmark(sample_inputs)
print(benchmarks_base)

# run sparse benchmarking
engine_sparse = compile_model(
    model="zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate", 
    batch_size=batch_size,
)
if not engine_sparse.cpu_vnni:
    print("WARNING: VNNI instructions not detected, quantization speedup not well supported")
benchmarks_sparse = engine_sparse.benchmark(sample_inputs)
print(benchmarks_sparse)

print(f"Speedup: {benchmarks_sparse.items_per_second / benchmarks_base.items_per_second:.2f}x")
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

For a more in-depth read on available APIs and workflows, check out the [examples](https://github.com/neuralmagic/deepsparse/blob/main/examples/) and [DeepSparse Engine documentation](https://docs.neuralmagic.com/deepsparse).

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

Then if you want to explore the [examples](https://github.com/neuralmagic/deepsparse/blob/main/examples/), clone the repository and any install additional dependencies found in example folders.

## Notebooks

For some step-by-step examples, we have Jupyter [notebooks](https://github.com/neuralmagic/deepsparse/blob/main/notebooks/) showing how to compile models with the DeepSparse Engine, check the predictions for accuracy, and benchmark them on your hardware.
  
## Available Models and Recipes

A number of pre-trained baseline and recalibrated models models in the [SparseZoo](https://github.com/neuralmagic/sparsezoo) can be used with the engine for higher performance. The types available for each model architecture are noted in its [SparseZoo model repository listing](https://docs.neuralmagic.com/sparsezoo/models).

## Resources and Learning More

- DeepSparse Engine [Documentation](https://docs.neuralmagic.com/deepsparse/), [Notebooks](https://github.com/neuralmagic/deepsparse/blob/main/notebooks/), [Examples](https://github.com/neuralmagic/deepsparse/blob/main/examples/)
- [DeepSparse API](https://docs.neuralmagic.com/deepsparse/api/deepsparse)
- [Debugging and Optimizing Performance](https://docs.neuralmagic.com/deepsparse/debugging-optimizing/)
- [SparseML Documentation](https://docs.neuralmagic.com/sparseml/)
- [Sparsify Documentation](https://docs.neuralmagic.com/sparsify/)
- [SparseZoo Documentation](https://docs.neuralmagic.com/sparsezoo/)
- Neural Magic [Blog](https://www.neuralmagic.com/blog/), [Resources](https://www.neuralmagic.com/resources/), [Website](https://www.neuralmagic.com/)

## Contributing

We appreciate contributions to the code, examples, and documentation as well as bug reports and feature requests! [Learn how here](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md).

## Join the Community

For user help or questions about the DeepSparse Engine, use our [GitHub Discussions](https://www.github.com/neuralmagic/deepsparse/discussions/). Everyone is welcome!

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please email us at [learnmore@neuralmagic.com](mailto:learnmore@neuralmagic.com) or fill out this [form](http://neuralmagic.com/contact/).

## License

The project's binary containing the DeepSparse Engine is licensed under the [Neural Magic Engine License](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC).

Example files and scripts included in this repository are licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE) as noted.

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
