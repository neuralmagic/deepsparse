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

# Exploration for including the ViT Pose model into the DeepSparse pipeline
Source: https://github.com/ViTAE-Transformer/ViTPose

## Installation
Follow the instructions in the `ViTPose` README file. Note:

- installing one of the dependencies, `mmcv` takes a lot of time and may look often like it is stuck. Be patient (or run with `-v` if helps), it will eventually terminate successfully.
- after the setup completes, it is also advisable to downgrade the default torch version from `1.3` to `1.2` to avoid CUDA errors (as I am writing this, we are internally supporting `torch==1.2.1`)

## Export

Before running the onnx export script, (manually) install `timm`, `onnx` and `onnxruntime`. Then, launch the [export script](https://github.com/ViTAE-Transformer/ViTPose/blob/main/tools/deployment/pytorch2onnx.py):

```bash
python tools/deployment/pytorch2onnx.py /home/ubuntu/damian/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py /home/ubuntu/damian/ViTPose/vitpose-b.pth 
```
The first argument is a config file (for ViTpose B)  the second argument is the `.pth` checkpoint (weights). Both can be found on the main site of the repository:
<img width="876" alt="image" src="https://user-images.githubusercontent.com/97082108/203548069-7239c758-8332-4d1d-b4d4-94774a4fcdef.png">

The resulting model is about 400mb. 

## Benchmarking in DeepSparse:

Naive benchmarking shows that for the dense model, the engine is roughly x2 faster than ORT:
<img width="1116" alt="Zrzut ekranu 2022-11-23 o 13 06 23" src="https://user-images.githubusercontent.com/97082108/203562298-3a96c653-58ef-4471-ab4a-faeb222c24b3.png">

## Postprocessing
ViT-Pose might be our first candidate for a "composed" deepsparse pipeline. 
It is a top-down pose estimation approach i.e. we first detect `n` people in the image, and then we estimate their poses individually (using bounding-box-cropped images).
We pass the cropped bounding boxes to ViT to get an array `(batch, no_keypoints, h, w)`. To decode this array, according to the original paper, 
we need some simple composition of transposed convolutions.

What I do naively for now: I  "squash" the array to `(h,w)` and then overlay it on the original image. We can see that the heatmap roughly coincides with the joints of the model.

<img width="585" alt="image" src="https://user-images.githubusercontent.com/97082108/204554128-a12deb08-6f6c-4383-aafc-ea5fee754e0e.png">

<img width="585" alt="image" src="https://user-images.githubusercontent.com/97082108/204554088-247296b0-21b1-43d2-928f-ae24cad9378a.png">

