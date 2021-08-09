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

# Ultralytics YOLO DeepSparse Inference Examples
This directory contains examples of benchmarking, annotating, and serving inferences
of YOLO models from the [ultralytics/yolov3](https://github.com/ultralytics/yolov3)
and [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
repositories using the DeepSparse Engine. The DeepSparse Engine is able to achieve
[real-time inferencing of YOLO on CPUs](https://neuralmagic.com/blog/benchmark-yolov3-on-cpus-with-deepsparse/)
by leveraging pruned and quantized YOLO models. These examples can load pre-trained,
sparsified models from [SparseZoo](https://github.com/neuralmagic/sparsezoo) or you can
create your own using the 
[Sparseml-Ultralytics-yolov5 integration](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/README.md).

## Installation
The dependencies for this example can be installed using `pip`:
```bash
pip3 install -r requirements.txt
```
Note: upgrade pip using `python -m pip install -U pip` before installing requirements
## SparseZoo Stubs
The following examples may be run with local ONNX YOLO models, or by using pre-trained, pre-sparsified YOLO models
from the [SparseZoo](https://sparsezoo.neuralmagic.com/).

The [SparseZoo](https://sparsezoo.neuralmagic.com/) contains both 
baseline and models sparsified to different degrees of inference performance vs. baseline loss recovery.

Each model in the SparseZoo has a specific stub that identifies it; any YOLO model stub in the SparseZoo can be used to
run the following examples.


| Model Name     |      Stub      | Description |
|----------|-------------|-------------|
| yolov5l-pruned | zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98 | Sparse YOLOv5l model trained with full FP32 precision that recovers 98% of its baseline mAP |
| yolov5l-pruned_quant | zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95 | Sparse INT8 quantized YOLOv5l model that recovers 95% of its baseline mAP |
| yolov5s-pruned | zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96 | Sparse YOLOv5l model trained with full FP32 precision that recovers 96% of its baseline mAP |
| yolov5s-pruned_quant | zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 | Sparse INT8 quantized YOLOv5s model that recovers 94% of its baseline mAP |
| yolov3-pruned | zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive_97 | Sparse YOLOv3 model trained with full FP32 precision that recovers 97% of its baseline mAP |
| yolov3-pruned_quant | zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 | Sparse INT8 quantized YOLOv3 model that recovers 94% of its baseline mAP |
| yolov5l-base | zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/base-none | Dense full precision YOLOv5l model |
| yolov5s-base | zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none | Dense full precision YOLOv5s model |
| yolov3-base | zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/base-none | Dense full precision YOLOv3-SPP model |


## Annotation Example
`annotate.py` is a script for using YOLO sparsified (or non-sparsified) models
to run inferences on images, videos, or webcam streams. For a full list of options
`python annotate.py -h`.

To run pruned-quantized YOLOv5s on a local webcam run:
```bash
python annotate.py \
    zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --source 0 \
    --quantized-inputs \
    --image-shape 416 416 \
    --no-save  # webcam only
```

In addition to webcam, `--source` can take a path to a `.jpg` file, directory or glob path
of `.jpg` files, or path to a `.mp4` video file.  If source is an integer and no
corresponding webcam is available, an exception will be raised.


## Benchmarking Example
`benchmark.py` is a script for benchmarking sparsified and quantized YOLO
performance with DeepSparse.  For a full list of options run `python benchmark.py -h`.

To run a YOLOv3 pruned-quantized benchmark run:
```bash
python benchmark.py \
    zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --batch-size 1 \
    --quantized-inputs
```

To run a YOLOv5s pruned-quantized benchmark run:
```bash
python benchmark.py \
    zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --batch-size 1 \
    --quantized-inputs
```

Note for quantized performance, your CPU must support VNNI instructions.
Review `/proc/cpuinfo` for the flag `avx512_vnni` to verify chipset compatibility.



## Example YOLO DeepSparse Flask Server

To illustrate how the DeepSparse Engine can be used for YOLO model deployments, this directory
contains a sample model server and client. 

The server uses Flask to create an app with the DeepSparse Engine hosting a
compiled YOLO model.
The client can make requests into the server returning object detection results for given images.

### Server

First, start up the host `server.py` with your model of choice, SparseZoo stubs are
also supported.

Example YOLOv3 Pruned Quantized command:
```bash
python server.py \
    zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --quantized-inputs
```

Example YOLOv5s Pruned Quantized command:
```bash
python server.py \
    zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --quantized-inputs
```

You can leave that running as a detached process or in a spare terminal.

This starts a Flask app with the DeepSparse Engine as the inference backend, accessible at `http://0.0.0.0:5543` by default.

The app exposes HTTP endpoints at:
- `/info` to get information about the compiled model
- `/predict` to send images to the model and receive as detected in response.
    The number of images should match the compiled model's batch size.

http://0.0.0.0:5543 is the default url and the user can also specify their own url while running the server.
For a full list of options, run `python server.py -h`.

Currently, the server is set to do pre-processing for the yolov3-spp
model; if other models are used, the image shape, output shapes, and
anchor grids should be updated. 

### Client

`client.py` provides a `YoloDetectionClient` object to make requests to the server easy.
The file is self-documented.  See example usage below:

```python
from client import YoloDetectionClient

remote_model = YoloDetectionClient()
image_path = "/PATH/TO/EXAMPLE/IMAGE.jpg"

model_outputs = remote_model.detect(image_path)
```
