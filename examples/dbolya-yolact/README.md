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

# YOLACT DeepSparse Inference Examples
This directory contains examples of benchmarking, annotating using YOLACT segmentation models from the [dbolya/yolact](https://github.com/dbolya/yolact) repositories using the DeepSparse Engine. 
The DeepSparse Engine achieves [real-time inferencing of YOLACT on CPUs]() by leveraging sparse-quantized YOLACT models. 
These examples can load pre-trained, sparsified models from [SparseZoo](https://github.com/neuralmagic/sparsezoo) or a custom-trained model created using the [SparseML YOLACT integration](https://github.com/neuralmagic/sparseml/blob/main/integrations/yolact/README.md).

## Installation
The [Neural Magic YOLACT Fork](https://github.com/neuralmagic/yolact) is modified to make annotation and benchmarking using the DeepSparse Engine easier. To begin, run the following command in the root directory of this example (`cd examples/dbolya-yolact`).

```bash
bash setup_integration.sh
```

Note: if you run into issues, try upgrading pip using `python -m pip install -U pip` before running the setup. 
We also recommend creating a `virtualenv` to keep project dependencies isolated.

## SparseZoo Stubs
The following examples may be run with local ONNX YOLACT models or by using pre-trained, pre-sparsified models
from the [SparseZoo](https://sparsezoo.neuralmagic.com/).

[SparseZoo](https://sparsezoo.neuralmagic.com/) contains both 
baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery.

Each model in the SparseZoo has a specific stub that identifies it; any YOLACT model stub in the SparseZoo can be used to
run the following examples.


| Model Name     |      Stub      | Description |
|----------|-------------|-------------|
| yolact-base |  | Dense full precision YOLACT model with a DarkNet-53 backbone|


## Annotation Example
`eval.py` inside the yolact repository has been updated for using YOLACT sparsified (or non-sparsified) models
to run inferences on images, videos, or webcam streams. For a full list of options
`python eval.py -h`.

To run image segmentation using YOLACT with DeepSparse on a local webcam run:
```bash
python eval.py \
    --trained_model PATH_OR_STUB_TO_YOLACT_ONNX \
    --source 0 
```

In addition to a webcam, `--source` can take a path to a `.jpg` file, directory or glob path
of `.jpg` files, or path to a `.mp4` video file.  If source is an integer and no
corresponding webcam is available, an exception will be raised.


## Benchmarking Example
`eval.py` can also be used to benchmark sparsified and quantized YOLACT
performance with DeepSparse.  For a full list of options run `python eval.py -h`.

To run a benchmark simply pass in the `--benchmark` flag while calling the script:
```bash
python eval.py \
    --trained_model PATH_OR_STUB_TO_YOLACT_ONNX \
    --source PATH_TO_IMAGE_DIRECTORY \
    --benchmark
```

Note: for quantized performance, your CPU must support VNNI instructions.
Review `/proc/cpuinfo` for the flag `avx512_vnni` to verify chipset compatibility.

## Citation
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
