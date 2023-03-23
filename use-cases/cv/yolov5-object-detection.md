# Deploying YOLOv5 Object Detection Models with DeepSparse

This page explains how to benchmark and deploy a YOLOv5 object detection model with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing
and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving
endpoint running DeepSparse with a single CLI.

## Installation Requirements

This use case requires the installation of [DeepSparse Server](/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](/user-guide/deepsparse-engine/hardware-support).

## Benchmarking

We can use the benchmarking utility to demonstrate the DeepSparse's performance. We ran the numbers below on a 12-core server.

### ONNX Runtime Baseline

As a baseline, let's check out ONNX Runtime's performance on YOLOv5. Make sure you have ORT installed (`pip install onnxruntime`).
```bash
deepsparse.benchmark \
  zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
  -b 64 -s sync -nstreams 1 \
  -e onnxruntime
> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 15.1734
> Latency Mean (ms/batch): 4217.1713
> Latency Median (ms/batch): 4088.7618
> Latency Std (ms/batch): 274.9809
> Iterations: 3
```
ONNX Runtime achieves 15 items/second with batch 64.
### DeepSparse Speedup
Now, let's run DeepSparse on an inference-optimized sparse version of YOLOv5 . This model has been 94% pruned-quantized, while retaining >99% accuracy of the dense baseline on the `coco` dataset.
```bash
!deepsparse.benchmark \
  zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
  -b 64 -s sync -nstreams 1 \
  -e deepsparse
> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 237.4027
> Latency Mean (ms/batch): 269.5658
> Latency Median (ms/batch): 268.4632
> Latency Std (ms/batch): 3.4354
> Iterations: 38
```
DeepSparse achieves 237 items/second, a 16x speed-up over ONNX Runtime!
## DeepSparse Engine
Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended using the Pipeline API but Engine is available if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 98% pruned-quantized YOLOv5 trained on `coco` from SparseZoo:
```python
from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98"
batch_size = 1
bert_engine = compile_model(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)
# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = bert_engine(inputs)
print(output)
# [array([[[5.54789925e+00, 4.28643513e+00, 9.98156166e+00, ...,
# ...
# -6.13238716e+00, -6.80812788e+00, -5.50403357e+00]]]]], dtype=float32)]
```
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

Let's start by downloading a sample image: 
```bash 
wget https://huggingface.co/spaces/neuralmagic/cv-yolo/resolve/main/Fruits.png
```
We will use the `Pipeline.create()` constructor to create an instance of an object detection Pipeline with a 96% pruned version of YOLOv5 trained on `coco`. We can then pass images to the `Pipeline` and receive the predictions. All the pre-processing (such as resizing the images) is handled by the `Pipeline`.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96"
yolo_pipeline = Pipeline.create(
  task="yolo",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
)
images = ["Fruits.png"]

# run inference on image file
pipeline_outputs = yolo_pipeline(images=images,  conf_thres=0.001)
print(len(pipeline_outputs.boxes[0]))
print(len(pipeline_outputs.scores[0]))
print(len(pipeline_outputs.labels[0]))
# 135
# 135
# 135
```
### Use Case Specific Arguments
The Object Detection Pipeline contains additional arguments for configuring a `Pipeline`.

#### IOU Threshold 
In the example below, we define a `iou_thres` of 0.6. You can adjust this depending on your use case. 

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96"
yolo_pipeline = Pipeline.create(
  task="yolo",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
    batch_size = 3
)
images = ["Fruits.png"] * 3

# run inference on image file
pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
print(len(pipeline_outputs.boxes[0]))
print(len(pipeline_outputs.scores[0]))
print(len(pipeline_outputs.labels[0]))
# 300
# 300
# 300
```
### Cross Use Case Functionality
Check out the [Pipeline User Guide](/user-guide/deepsparse/deepsparse-pipelines) for more details on configuring a Pipeline.
## DeepSparse Server
Built on the popular FastAPI and Uvicorn stack, DeepSparse Server enables you to set up a REST endpoint for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all the utilities provided by Pipelines.

The CLI command below launches an object detection pipeline with a 94% pruned-quantized YOLOv5 model:

```bash
deepsparse.server task yolo --model_path "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94" --port 5543
```
```python
import requests
import json

url = 'http://0.0.0.0:5543/predict/from_files'
path = ['pets.jpg'] # list of images for inference
files = [('request', open(img, 'rb')) for img in path]
resp = requests.post(url=url, files=files)
annotations = json.loads(resp.text) # dictionary of annotation results
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
print(labels)
# [['16.0', '16.0', '16.0', '15.0', '15.0']]
```
#### Use Case Specific Arguments
To use the `class_names` argument, create a Server configuration file for passing the argument via kwargs.

This configuration file sets `class_names` to `coco`:
```yaml
# yolo-config.yaml
endpoints:
  - task: yolo
    model: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
    kwargs:
      class_names: coco
```
Start the server: 
```bash 
deepsparse.server --config-file yolo-config.yaml
```
Run inference: 
```python
import requests
import json

url = 'http://0.0.0.0:5555/predict/from_files'
path = ['pets.jpg'] # list of images for inference
files = [('request', open(img, 'rb')) for img in path]
resp = requests.post(url=url, files=files)
annotations = json.loads(resp.text) # dictionary of annotation results
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
print(labels)
# [['dog', 'dog', 'dog', 'cat', 'cat']]
```
### Cross Use Case Functionality

Check out the [Server User Guide](/user-guide/deepsparse/deepsparse-server) for more details on configuring the Server.