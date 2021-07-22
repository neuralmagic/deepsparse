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

# Image Classification Example

This directory holds example scripts and notebooks for benchmarking and serving image classification models using the [DeepSparse Engine](https://docs.neuralmagic.com/deepsparse/index.html).
These examples can load pre-trained, sparsified models from [SparseZoo](https://github.com/neuralmagic/sparsezoo) 
or you can specify your own [ONNX](https://github.com/onnx/onnx) file.

## Installation
The dependencies for this example can be installed using `pip` and the supplied `requirements.txt` file, make sure to upgrade `pip` using `python -m pip install -U pip
` before running the following:
```bash
pip3 install -r requirements.txt
```

## Notebook

There is a step-by-step [classification.ipynb notebook](https://github.com/neuralmagic/deepsparse/blob/main/examples/classification/classification.ipynb) for this example.

## Execution

Example command for running a `mobilenet_v2` model with batch size 8 and 4 cores used:
```bash
python classification.py mobilenet_v2 --batch_size 8 --num_cores 4
```

Run with the `-h` flag to see all available models.

## Benchmarking Example
`benchmark.py` is a script for benchmarking sparsified image classification model performance with DeepSparse. For a full list of options run `python benchmark.py -h`.
To run a benchmark using the DeepSparse Engine with a moderately pruned ResNet model that uses all available CPU cores and batch size 1, run:

```bash
python benchmark.py "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate"
```

Example using own model and data:

```bash
python benchmark.py path_to_onnx_file \
--data-path path_to_data_directory
```

Replace `path_to_onnx_file` with [SparseZoo](http://sparsezoo.neuralmagic.com/) stub or filepath to ONNX model file and replace `path_to_data_directory` with appropriate SparseZoo stub or path to a directory containing `.npz` files to be used as model inputs.
## Example Image Classification DeepSparse Flask Deployment

The [server.py](https://github.com/neuralmagic/deepsparse/blob/main/examples/classification/server.py) script uses Flask for hosting a compiled image classification model with the DeepSparse Engine.
Client can make requests into the server returning inference results for given inputs.

### Server
First, start up the host server.py with your model of choice, SparseZoo stubs are also supported.

Example command:
```bash
python server.py "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate"
```

You can leave that running as a detached process or in a spare terminal.

This starts a Flask app with the DeepSparse Engine as the inference backend, accessible at http://0.0.0.0:5543 by default.

Once the server is running, our app creates two URLs, exposing the following HTTP endpoints:

- `http://0.0.0.0:5543/info` to get information about the compiled model
- `http://0.0.0.0:5543/predict` to post inputs to the model and get inference results. The number of inputs should match the compiled model's batch size.

Here http://0.0.0.0:5543 is the default url and port number, user can also specify their own url while running [server.py](https://github.com/neuralmagic/deepsparse/blob/main/examples/classification/server.py)
For a full list of options, run `python server.py -h`.
  
  
### Client

`client_examples.py` provides a `Client` object to make requests to the 
server easy.
The file is self-documented.  See example usage below:

```python
from client_example import Client
from helper import load_data, _BatchLoader

client = Client()
resnet_stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate"
batch_loader = _BatchLoader(data=load_data(resnet_stub), batch_size=1, iterations=1)

for batch in batch_loader:
    out = client.classify(images=batch, )
    print(f"outputs:{out}")
```

### SparseZoo Stubs
[SparseZoo](http://sparsezoo.neuralmagic.com/) is a constantly growing repository of sparsified (pruned and 
pruned-quantized) models with matching sparsification recipes for neural networks. It simplifies and accelerates your time to value in building performant deep learning models with a collection of inference-optimized models and recipes to prototype.

Available via API and hosted in the cloud, the [SparseZoo](http://sparsezoo.neuralmagic.com/) contains both 
baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery. Recipe-driven approaches built around sparsification algorithms allow you to take the models as given, transfer-learn from the models onto private datasets, or transfer the recipes to your architectures.

Each model in the SparseZoo has a specific stub that identifies it. The stubs are made up of the following structure:

`DOMAIN/SUB_DOMAIN/ARCHITECTURE{-SUB_ARCHITECTURE}/FRAMEWORK/REPO/DATASET{-TRAINING_SCHEME}/SPARSE_NAME-SPARSE_CATEGORY-{SPARSE_TARGET}`

Learn more at 
- [SparseZoo Docs](https://docs.neuralmagic.com/sparsezoo/)
- [SparseZoo Website](https://sparsezoo.neuralmagic.com/) 
- [SparseZoo GitHub Repository](https://github.com/neuralmagic/sparsezoo)

A few classification models from [SparseZoo](https://sparsezoo.neuralmagic.com/):

| Model Name     |      Stub      | Description |
|----------|-------------|-------------|
|resnet-base|zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none|This model is the base [ResNet-50](https://arxiv.org/abs/1512.03385) model|
| resnet-pruned-moderate | zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate |This model is a sparse, [ResNet-50](https://arxiv.org/abs/1512.03385) model that achieves 99% of the accuracy of the original baseline model (76.1% top1). Pruned layers achieve 88% sparsity.|
|resnet-pruned-conservative|zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-conservative|This model is a sparse, [ResNet-50](https://arxiv.org/abs/1512.03385) model that achieves full recovery original baseline model accuracy (76.1% top1). Pruned layers achieve 80% sparsity.|
|resnet-pruned-quantized-moderate|zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate|This model is a sparse, INT8 quantized ResNet-50 model that achieves 99% of the original baseline accuracy. The average model sparsity is about 72% with pruned layers achieving between 70-80% sparsity. This pruned quantized model achieves 75.46% top1 accuracy on the ImageNet dataset.|
|mobilenetv1-base|zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none|Standard [MobileNet](https://arxiv.org/abs/1704.04861) architecture with width=1.0 trained on the ImageNet dataset. Baseline model: 70.9% for validation top1 accuracy|
|mobilenetv1-pruned-quantized|zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate|Pruned [MobileNet](https://arxiv.org/abs/1704.04861) architecture with width=1.0 trained on the ImageNet dataset. Recalibrated performance model: within 99% of baseline validation top1 accuracy (70.9%)|
|mobilenetv1-pruned-moderate|zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate|This model is a sparse, [MobileNetV1](https://arxiv.org/abs/1704.04861) model that achieves 99% of the accuracy of the original baseline model (70.9% top1). Pruned layers achieve between 70-90% sparsity.|
|mobilenetv1-pruned-conservative|zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-conservative|This model is a sparse, [MobileNetV1](https://arxiv.org/abs/1704.04861) model that achieves the same accuracy as the original baseline model. Pruned layers achieve between 60-86% sparsity. This pruned quantized model achieves 70.9% top1 accuracy on the ImageNet dataset.|
