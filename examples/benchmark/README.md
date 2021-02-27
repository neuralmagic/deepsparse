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

# Benchmarking Examples

This directory holds examples for comparing inference on ONNX models, both for performance and correctness.

## Installation

Install DeepSparse with `pip install deepsparse` and the additional external requirements with `pip install -r requirements.txt`.

## Execution

### ResNet-50 Benchmark

`resnet50_benchmark.py` is a script for benchmarking all of the sparsified ResNet50 V1 models hosted on SparseZoo, on the DeepSparse engine.

Example command for ResNet50 benchmarks with batch size 128:
```bash
python resnet50_benchmark.py --batch_size 128
```

### ONNX Benchmark

`run_benchmark.py` is a script for benchmarking an ONNX model over random inputs and using both the DeepSparse Engine and ONNXRuntime, comparing results.

Example command for benchmarking a downloaded resnet50 model for batch size 8, over 100 iterations:
```bash
python run_benchmark.py ~/Downloads/resnet50.onnx --batch_size 8 --num_iterations 100
```

### Check Correctness

`check_correctness.py` is a script for generating random input from an ONNX model and running the model both through the DeepSparse Engine and ONNXRuntime, comparing outputs to confirm they are the same.

Example command for checking a downloaded resnet50 model for batch size 8:
```bash
python check_correctness.py resnet50.onnx --batch_size 8
```
