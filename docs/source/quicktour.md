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

## Quick Tour

To expedite inference and benchmarking on real models, we include the `sparsezoo` package. 
[SparseZoo](https://github.com/neuralmagic/sparsezoo) hosts inference-optimized models, 
trained on repeatable sparsification recipes using state-of-the-art techniques from 
[SparseML](https://github.com/neuralmagic/sparseml).

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

Let's say that we want a version of ResNet-50 that recovers close to the baseline and is very performant, we can choose the pruned_quant-moderate model.
This model will run [nearly 7 times faster](linktoresnet50example) than the baseline model on a compatible CPU (VNNI instruction set enabled).
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
