..
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

====================
DeepSparse |version|
====================

CPU inference engine that delivers unprecedented performance for sparse models.

.. raw:: html

    <div style="margin-bottom:16px;">
        <a href="https://github.com/neuralmagic/deepsparse/blob/master/LICENSE">
            <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/deepsparse.svg?color=purple&style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://docs.neuralmagic.com/deepsparse/index.html">
            <img alt="Documentation" src="https://img.shields.io/website/http/neuralmagic.com/deepsparse/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://github.com/neuralmagic/deepsparse/releases">
            <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/deepsparse.svg?style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://github.com/neuralmagic.com/deepsparse/blob/master/CODE_OF_CONDUCT.md">
            <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25 style="margin-bottom:4px;">
        </a>
         <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
            <img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=25 style="margin-bottom:4px;">
        </a>
         <a href="https://medium.com/limitlessai">
            <img src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height=25 style="margin-bottom:4px;">
        </a>
        <a href="https://twitter.com/neuralmagic">
            <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25 style="margin-bottom:4px;">
        </a>
     </div>

Overview
========

The DeepSparse Engine is a CPU runtime that delivers unprecedented performance by taking advantage of
natural sparsity within neural networks to reduce compute required as well as accelerate memory bound workloads.
It is focused on model deployment and scaling machine learning pipelines,
fitting seamlessly into your existing deployments as an inference backend.

This repository includes package APIs along with examples to quickly get started learning about and
actually running sparse models.

Compatability
=============

The DeepSparse Engine ingests models in the `ONNX <https://onnx.ai/ />`_ format,
allowing for compatibility with `PyTorch <https://pytorch.org/docs/stable/onnx.html />`_,
`TensorFlow <https://github.com/onnx/tensorflow-onnx />`_, `Keras <https://github.com/onnx/keras-onnx />`_,
and `many other frameworks <https://github.com/onnx/onnxmltools />`_ that support it.
This reduces the extra work of preparing your trained model for inference to just one step of exporting.

Related Products
================

- `Sparse Zoo <https://github.com/neuralmagic/sparsezoo />`_:
  Neural network model repository for highly sparse models and optimization recipes
- `SparseML <https://github.com/neuralmagic/sparseml />`_:
  Libraries for state-of-the-art deep neural network optimization algorithms,
  enabling simple pipelines integration with a few lines of code
- `Sparsify <https://github.com/neuralmagic/sparsify />`_:
  Easy-to-use autoML interface to optimize deep neural networks for
  better inference performance and a smaller footprint

Resources and Learning More
===========================

- `SparseZoo Documentation <https://docs.neuralmagic.com/sparsezoo/ />`_
- `SparseML Documentation <https://docs.neuralmagic.com/sparseml/ />`_
- `Sparsify Documentation <https://docs.neuralmagic.com/sparsify/ />`_
- `DeepSparse Documentation <https://docs.neuralmagic.com/deepsparse/ />`_
- `Neural Magic Blog <https://www.neuralmagic.com/blog/ />`_,
  `Resources <https://www.neuralmagic.com/resources/ />`_,
  `Website <https://www.neuralmagic.com/ />`_

Release History
===============

Official builds are hosted on PyPi
- stable: `deepsparse <https://pypi.org/project/deepsparse/ />`_
- nightly (dev): `deepsparse-nightly <https://pypi.org/project/deepsparse-nightly/ />`_

Additionally, more information can be found via
`GitHub Releases <https://github.com/neuralmagic/deepsparse/releases />`_.

.. toctree::
    :maxdepth: 3
    :caption: General

    quicktour
    installation
    hardware

.. toctree::
    :maxdepth: 3
    :caption: Performance

    debugging-optimizing/index

.. toctree::
    :maxdepth: 2
    :caption: API

    api/deepsparse
