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

# Examples

This directory contains self-documented examples to illustrate how to make use of the DeepSparse Engine. 

For instructions on how to run each example, either check the script header or run them with `-h`.

Open a Pull Request to [contribute](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md) your own examples.

## Important note

To run these scripts, you may need to install some packages for specific examples and ensure you have the correct release of `deepsparse` installed.

In a new virtual environment:
```base
pip install deepsparse
```
Then cd in the example folder of your choice and run
```bash
pip install -r requirements.txt
```

## Examples

| Notebook     |      Description      |
|----------|-------------|
| [Benchmark and ONNX Model Correctness](benchmark/)  | Comparing predictions and benchmark performance between DeepSparse Engine and ONNXRuntime  |
| [Classification](classification/)  | How to use classification models from SparseZoo to inference and benchmark with the DeepSparse Engine  |
| [Detection](detection/)  | How to use object detection models from SparseZoo to inference and benchmark with the DeepSparse Engine  |
| [Model Server](flask/)  | Simple model server and client example, showing how to use the DeepSparse Engine as an inference backend for a real-time inference server |
