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

# Object Detection Example

This directory holds an example for downloading an object detection model from SparseZoo with real data and using the DeepSparse Engine for inference and benchmarking.

## Installation

Install DeepSparse with `pip install deepsparse`.

## Notebook

There is a step-by-step [detection.ipynb notebook](https://github.com/neuralmagic/deepsparse/blob/main/examples/detection/detection.ipynb) for this example.

## Execution

Example command for running a `yolo_v3` model with batch size 8 and 4 cores:
```bash
python detection.py yolo_v3 --batch_size 8 --num_cores 4
```
