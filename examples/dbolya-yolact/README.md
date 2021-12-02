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
The DeepSparse Engine achieves real-time inferencing of YOLACT on CPUs by leveraging sparse-quantized [YOLACT](https://arxiv.org/abs/1904.02689) models. 
These examples can load pre-trained, sparsified models from [SparseZoo](https://sparsezoo.neuralmagic.com/) or a custom-trained model created using the 
[SparseML YOLACT integration](https://github.com/neuralmagic/sparseml/blob/main/integrations/dbolya-yolact/README.md).

## Installation
The [Neural Magic YOLACT Fork](https://github.com/neuralmagic/yolact) is modified to make annotation and benchmarking using the DeepSparse Engine easier. 
To begin, run the following command in the root directory of this example (`cd examples/dbolya-yolact`).

```bash
bash setup_integration.sh
```

Note: if you run into issues, try upgrading pip using `python -m pip install -U pip` before running the setup. 
We also recommend creating a `virtualenv` to keep project dependencies isolated.

## SparseZoo Stubs

The following examples may be run with local ONNX YOLACT models or by using pre-trained, pre-sparsified models
from the [SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=segmentation&page=1).

[SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=segmentation&page=1) contains both 
baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery.

Each model in the SparseZoo has a specific stub that identifies it; any YOLACT model stub in the SparseZoo can be used to
run the following examples.

| Sparsification Type | Description                                                                       | Zoo Stub                                                                     | COCO mAP@all | Size on Disk | DeepSparse Performance** |
|---------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------|--------------|--------------|--------------------------|
| Baseline            | The baseline, pretrained model on the COCO dataset.                               | zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none           | 0.288        | 170 MB       | -- img/sec               |
| Pruned              | A highly sparse, FP32 model that recovers close to the baseline model.            | zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned90-none       | 0.286        | 30.1 MB      | -- img/sec               |
| Pruned Quantized    | A highly sparse, INT8 model that recovers reasonably close to the baseline model. | zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none | 0.282        | 9.7 MB       | -- img/sec               |

## Annotation Example

`eval.py` inside the `yolact` repository has been updated for using YOLACT sparsified (or non-sparsified) models
to run inferences on images, videos, or webcam streams. For a full list of options
`python eval.py -h`.

To run image segmentation using YOLACT with DeepSparse on a local webcam run:
```bash
python eval.py \
    --trained_model PATH_OR_STUB_TO_YOLACT_ONNX \
    --video 0 
```


In addition to a webcam, `eval.py` can take a path to a `.jpg` file, directory, 
or path to a `.mp4` video file.  If the source is an integer and no
 corresponding webcam is available, an exception will be raised.

Example commands are as follows:

```bash
# Annotate an Image using DeepSparse
# python eval.py --trained_model PRETRAINED_ONNX_WEIGHTS \
# --image input.png:output.png

python eval.py --trained_model weights/model.onnx \
--image data/yolact_example_0.png:data/yolact_example_out_0.png

# Annotate Images using DeepSparse
# python eval.py --trained_model PRETRAINED_ONNX_WEIGHTS \
# --images input_dir:output_dir --score SCORE_THRESHOLD

python eval.py --trained_model weights/model.onnx \
--images input_dir:output_dir --score 0.6

# Annotate Video using DeepSparse
# python eval.py --trained_model PRETRAINED_ONNX_WEIGHTS \
# --video input.mp4:output.mp4 --score 0.6

python eval.py --trained_model weights/model.onnx \
--video input.mp4:output.mp4 --score 0.6

# Annotate Webcam using DeepSparse
# python eval.py --trained_model PRETRAINED_ONNX_WEIGHTS \
# --video 0 --score 0.6

python eval.py --trained_model weights/model.onnx \
--video 0 --score 0.6
```

## Benchmarking Example
`eval.py` can also be used to benchmark sparsified and quantized YOLACT
performance with DeepSparse on the [COCO](https://cocodataset.org/#home) validation set.  For a full list of options run `python eval.py -h`.

First, set up the validation data by running `bash data/scripts/COCO.sh` inside the `yolact` folder. 
Then, to run a benchmark simply pass in the `--benchmark` flag while calling the script:
```bash
# Using DeepSparse and the entire COCO validation set
# python eval.py --trained_model PRETRAINED_ONNX_WEIGHTS 
python eval.py --trained_model weights/model.onnx --score 0.6 --benchmark

# Using DeepSparse and the entire COCO validation set,
# using a valid YOLACT SparseZoo stubs
# python eval.py --trained_model SPARSEZOO_STUB \
# --benchmark
python eval.py \
--trained_model zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none \
--score 0.6 \
--batch_size 1 \
--benchmark

# Using DeepSparse on the COCO validation set for fixed num of iterations
# python eval.py --trained_model PRETRAINED_ONNX_WEIGHTS \
# --score SCORE_THRESHOLD --warm_up_iterations WARM_UP_ITERATIONS \
# --num_iterations NUM_ITERATIONS \
# --benchmark
python eval.py \
--trained_model weights/model.onnx \
--score 0.6  \
--warm_up_iterations 10 \
--num_iterations 100 \
--benchmark
```

The average fps (frames per second) will be calculated and displayed on the console. 
Note: for quantized performance, your CPU must support VNNI instructions.
Review `/proc/cpuinfo` for the flag `avx512_vnni` to verify chipset compatibility.

## Citation
```bibtex
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```
