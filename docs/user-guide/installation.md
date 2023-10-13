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

# DeepSparse Installation

DeepSparse is tested on Python 3.8-3.11, ONNX 1.5.0-1.15.0, ONNX opset version 11+ and is [manylinux compliant](https://peps.python.org/pep-0513/).

It currently supports Intel and AMD AVX2, AVX-512, and VNNI x86 instruction sets.

## General Install

Use the following command to install DeepSparse with pip:

```bash
pip install deepsparse
```

## Installing the Server

DeepSparse Server allows you to serve models and pipelines through an HTTP interface using the `deepsparse.server` CLI.
To install, use the following extra option:

```bash
pip install deepsparse[server]
```

## Installing YOLO

The Ultralytics YOLOv5 models require extra dependencies for deployment. To use YOLO models, install with the following extra option:

```bash
pip install deepsparse[yolo]         # just yolo requirements
pip install deepsparse[yolo,server]  # both yolo + server requirements
```
