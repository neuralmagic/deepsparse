---
title: "DeepSparse Community"
metaTitle: "DeepSparse Community Installation"
metaDescription: "Installation instructions for DeepSparse enabling performant neural network deployments"
index: 1000
---

# DeepSparse Community Installation

[DeepSparse Community](/products/deepsparse) enables GPU-class performance on commodity CPUs.

Currently, DeepSparse is tested on Python 3.7-3.10, ONNX 1.5.0-1.10.1, ONNX opset version 11+ and is [manylinux compliant](https://peps.python.org/pep-0513/).

We currently support x86 CPU architectures.

DeepSparse is available in two versions:
1. [**DeepSparse Community**](/products/deepsparse) is free for evaluation, research, and non-production use with our [DeepSparse Community License](https://neuralmagic.com/legal/engine-license-agreement/).
2. [**DeepSparse Enterprise**](/products/deepsparse-ent) requires a Trial License or [can be fully licensed](https://neuralmagic.com/legal/master-software-license-and-service-agreement/) for production, commercial applications.

## General Install

Use the following command to install DeepSparse Community with pip:

```bash
pip install deepsparse
```

## Installing the Server

[DeepSparse Server](/user-guide/deploying-deepsparse/deepsparse-server) allows you to serve models and pipelines through an HTTP interface using the deepsparse.server CLI.
To install, use the following extra option:

```bash
pip install deepsparse[server]
```

## Installing YOLO

The [Ultralytics YOLOv5](/use-cases/object-detection/deploying) models require extra dependencies for deployment.
To use YOLO models, install with the following extra option:

```bash
pip install deepsparse[yolo]         # just yolo requirements
pip install deepsparse[yolo,server]  # both yolo + server requirements
```
 
