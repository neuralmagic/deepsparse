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
[SparseZoo](https://github.com/neuralmagic/sparsezoo) hosts inference optimized models, 
trained on repeatable optimization recipes using state-of-the-art techniques from 
[SparseML](https://github.com/neuralmagic/sparseml).

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

When exploring available optimized models, you can use the `Zoo.search_optimized_models` 
utility to find models that share a base. 

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

Great. We can see there are two pruned versions targeting FP32, 
`conservative` at 100% and `moderate` at >= 99% of baseline accuracy. 
There is also a `pruned_quant` variant targeting INT8.

Let's say you want to evaluate best performance on FP32 and are okay with a small drop in accuracy, 
so we can choose `pruned-moderate` over `pruned-conservative`.

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
