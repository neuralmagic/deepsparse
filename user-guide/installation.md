# DeepSparse Installation

DeepSparse is tested on Python 3.7-3.10, ONNX 1.5.0-1.10.1, ONNX opset version 11+ and is [manylinux compliant](https://peps.python.org/pep-0513/). It currently supports Intel and AMD AVX2, AVX512, and VNNI x86 instruction sets.

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
