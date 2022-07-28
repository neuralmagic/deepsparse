# Image Classification Inference Pipelines


[DeepSparse] Image Classification integration allows accelerated inference, 
serving, and benchmarking of sparsified image classification models.
This integration allows for leveraging the DeepSparse Engine to run 
sparsified image classification inference with GPU-class performance directly 
on the CPU.

The DeepSparse Engine takes advantage of sparsity within neural networks to 
reduce compute as well as accelerate memory-bound workloads. 
The Engine is particularly effective when leveraging sparsification methods 
such as [pruning](https://neuralmagic.com/blog/pruning-overview/) and 
[quantization](https://arxiv.org/abs/1609.07061). These techniques result in 
significantly more performant and smaller models with limited to no effect on 
the baseline metrics.

## Getting Started

Before you start your adventure with the DeepSparse Engine, make sure that 
your machine is compatible with our [hardware requirements].

### Installation

```pip install deepsparse```

### Model Format

By default, to deploy image classification models using the DeepSparse Engine,
the model should be supplied in the [ONNX] format. 
This grants the Engine the flexibility to serve any model in a framework-agnostic
manner. 

Below we describe two possibilities to obtain the required ONNX model.

#### Exporting the onnx file from the contents of a local checkpoint

This pathway is relevant if you intend to deploy a model created using [SparseML] library. 
For more information refer to the appropriate integration documentation in [SparseML].

1. The output of the [SparseML] training is saved to output directory `/{save_dir}` (e.g. `/trained_model`)
2. Depending on the chosen framework, the model files are saved to `model_path`=`/{save_dir}/{framework_name}/{model_tag}` (e.g `/trained_model/pytorch/resnet50/`)
3. To generate an onnx model, refer to the [script for image classification ONNX export](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/image_classification/export.py).

Example:
```bash
sparseml.image_classification.export_onnx \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 \
    --checkpoint-path ~/checkpoints/resnet50_checkpoint.pth
```
This creates `model.onnx` file, in the parent directory of your `model_path`

####  Directly using the SparseZoo stub

Alternatively, you can skip the process of onnx model export by downloading all the required model data directly from Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com/).
Example:
```python
from sparsezoo import Model

# you can lookup an appropriate model stub here: https://sparsezoo.neuralmagic.com/
model_stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none"
model = Model(model_stub)

# directly download the model data to your local directory
model_path = model.path

# the onnx model file is there, ready for deployment
import os 
os.path.isfile(model.onnx_model.path)
>>>True
```


## Deployment APIs

DeepSparse provides both a python Pipeline API and an out-of-the-box model 
server that can be used for end-to-end inference in either existing python 
workflows or as an HTTP endpoint. Both options provide similar specifications 
for configurations and support a variety of Image Classification models.

### Python API

Pipelines are the default interface for running the inference with the 
DeepSparse Engine.

Once a model is obtained, either through [SparseML] training or directly from [SparseZoo],
`deepsparse.Pipeline` can be used to easily facilitate end to end inference and deployment
of the sparsified image classification model.

If no model is specified to the `Pipeline` for a given task, the `Pipeline` will automatically
select a pruned and quantized model for the task from the `SparseZoo` that can be used for accelerated
inference. Note that other models in the [SparseZoo] will have different tradeoffs between speed, size,
and accuracy.

To learn about sparsification in more detail, refer to [SparseML docs](https://docs.neuralmagic.com/sparseml/)

### HTTP Server

As an alternative to Python API, the DeepSparse inference server allows you to 
serve ONNX models and pipelines in HTTP. Both configuring and making requests 
to the server follow the same parameters and schemas as the Pipelines enabling 
simple deployment. Once launched, a `/docs` endpoint is created with full 
endpoint descriptions and support for making sample requests.

Example deployment using a 95% pruned resnet50 is given below
For full documentation on deploying sparse image classification models with the
DeepSparse Server, see the [documentation](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server).

##### Installation

The deepsparse server requirements can be installed by specifying the `server` 
extra dependency when installing DeepSparse.

```bash
pip install deepsparse[server]
```

## Deployment Use Cases

The following section includes example usage of the Pipeline and server APIs for
various image classification models. 

[List of Image Classification SparseZoo Models](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=classification&page=1)


#### Python Pipeline

```python
from deepsparse import Pipeline
cv_pipeline = Pipeline.create(
  task='image_classification', 
  model_path='zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none',  # Path to checkpoint or SparseZoo stub
)
input_image = "my_image.png" # path to input image
inference = cv_pipeline(images=input_image)
```

#### HTTP Server

Spinning up:
```bash
deepsparse.server \
    --task image_classification \
    --model_path "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none" \
    --port 5543
```

Making a request:
```python
import requests

url = 'http://0.0.0.0:5543/predict/from_files'
path = ['golfish.jpeg', 'golfish.jpeg'] # just put the name of images in here
files = [('request', open(img, 'rb')) for img in path]
resp = requests.post(url=url, files=files)
```

### Benchmarking

The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs. 
Want to find out how fast our sparse ONNX models perform inference? 
You can quickly do benchmarking tests on your own with a single CLI command!

You only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:
```bash
deepsparse.benchmark zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none
```
Output:
```bash
Original Model Path: zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none
Batch Size: 1
Scenario: async
Throughput (items/sec): 299.2372
Latency Mean (ms/batch): 16.6677
Latency Median (ms/batch): 16.6748
Latency Std (ms/batch): 0.1728
Iterations: 2995
```

To learn more about benchmarking, refer to the appropriate documentation.
Also, check out our [Benchmarking tutorial](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)!

## Tutorials:
For a deeper dive into using image classification models within the Neural Magic
ecosystem, refer to the detailed tutorials on our [website](https://neuralmagic.com/):
- [CV Use Cases](https://neuralmagic.com/use-cases/#computervision)

## Support
For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues).


[DeepSparse]: https://github.com/neuralmagic/deepsparse
[hardware requirements]: https://docs.neuralmagic.com/deepsparse/source/hardware.html
[ONNX]: https://onnx.ai/
[SparseML]: https://github.com/neuralmagic/sparseml
[SparseML Image Classification Documentation]: https://github.com/neuralmagic/sparseml/tree/main/src/sparseml/pytorch/image_classification/README_image_classification.md
[SparseZoo]: https://sparsezoo.neuralmagic.com/