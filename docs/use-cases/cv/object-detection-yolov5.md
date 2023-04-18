# Deploying YOLOv5 Object Detection Models with DeepSparse

This page explains how to benchmark and deploy a YOLOv5 object detection model with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving endpoint running DeepSparse with a single CLI.

This example uses YOLOv5s. For a full list of pre-sparsified object detection models, [check out the SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1).

## Installation Requirements

This use case requires the installation of [DeepSparse Server and YOLO](https://docs.neuralmagic.com/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](https://docs.neuralmagic.com/user-guides/deepsparse-engine/hardware-support).

## Benchmarking

We can use the benchmarking utility to demonstrate the DeepSparse's performance. The numbers below were run on a 4 core `c6i.2xlarge` instance in AWS.

### ONNX Runtime Baseline

As a baseline, let's check out ONNX Runtime's performance on YOLOv5s. Make sure you have ORT installed (`pip install onnxruntime`).

```bash
deepsparse.benchmark \
  zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none \
  -b 64 -s sync -nstreams 1 \
  -e onnxruntime

> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 12.2369
```
ONNX Runtime achieves 12 items/second with batch 64.

### DeepSparse Speedup
Now, let's run DeepSparse on an inference-optimized sparse version of YOLOv5s. This model has been 85% pruned and quantized.

```bash
deepsparse.benchmark \
  zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none \
  -b 64 -s sync -nstreams 1

> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 72.55
```
DeepSparse achieves 73 items/second, a 5.5x speed-up over ONNX Runtime!

## DeepSparse Engine
Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended using the Pipeline API but Engine is available if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 85% pruned-quantized YOLOv5s model from SparseZoo:

```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none"
batch_size = 1
compiled_model = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)
# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = compiled_model(inputs)

print(output[0].shape)
print(output[0])

# (1,25200, 85)
# [array([[[5.54789925e+00, 4.28643513e+00, 9.98156166e+00, ...,
# ...
# -6.13238716e+00, -6.80812788e+00, -5.50403357e+00]]]]], dtype=float32)]
```

## DeepSparse Pipeline
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

Let's start by downloading a sample image: 
```bash 
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

We will use the `Pipeline.create()` constructor to create an instance of an object detection Pipeline with a 85% pruned version of YOLOv5s trained on `coco`. We can then pass images to the `Pipeline` and receive the predictions. All the pre-processing (such as resizing the images and running NMS) is handled by the `Pipeline`.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none"
yolo_pipeline = Pipeline.create(
  task="yolo",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
)
images = ["basilica.jpg"]

# run inference on image file
pipeline_outputs = yolo_pipeline(images=images)
print(pipeline_outputs.boxes)
print(pipeline_outputs.labels)

# [[[262.56866455078125, 483.48693108558655, 514.8401184082031, 611.7606239318848], [542.7222747802734, 385.72591066360474, 591.0432586669922, 412.0340189933777], [728.4929351806641, 403.6355793476105, 769.6295471191406, 493.7961976528168], [466.83229064941406, 383.6878204345703, 530.7117462158203, 408.8705735206604], [309.2399597167969, 396.0068359375, 362.10223388671875, 435.58393812179565], [56.86535453796387, 409.39830899238586, 99.50672149658203, 497.8857614994049], [318.8877868652344, 388.9980583190918, 449.08460998535156, 587.5987024307251], [793.9356079101562, 390.5112290382385, 861.0441284179688, 489.4586777687073], [449.93934631347656, 441.90707445144653, 574.4951934814453, 539.5000758171082], [99.09783554077148, 381.93165946006775, 135.13665390014648, 458.19711089134216], [154.37461853027344, 386.8395175933838, 188.95138549804688, 469.1738815307617], [14.558289527893066, 396.7127945423126, 54.229820251464844, 487.2396695613861], [704.1891632080078, 398.2202727794647, 739.6305999755859, 471.5654203891754], [731.9091796875, 380.60836935043335, 761.627197265625, 414.56129932403564]]] << list of bounding boxes >>

# [['3.0', '2.0', '0.0', '2.0', '2.0', '0.0', '0.0', '0.0', '3.0', '0.0', '0.0', '0.0', '0.0', '0.0']] << list of label ids >>
```

### Use Case Specific Arguments
The YOLOv5 pipeline contains additional arguments for configuring a Pipeline.

#### Image Shape

DeepSparse runs with static shapes. By default, YOLOv5 inferences run with images of shape 640x640. The Pipeline accepts images of any size and scales the images to image shape specified by the ONNX graph.

We can override the image shape used by DeepSparse with the `image_size` argument. In the example below, we run the inferences at 320x320.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none"
yolo_pipeline = Pipeline.create(
  task="yolo",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  image_size=(320,320)
)
images = ["basilica.jpg"]

# run inference on image file
pipeline_outputs = yolo_pipeline(images=images)
print(pipeline_outputs.boxes)
print(pipeline_outputs.labels)
```

#### Class Names
We can specify class names for the labels by passing a dictionary. In the example below, we just use
the first 4 classes from COCO for the sake of a quick example.

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none"
yolo_pipeline = Pipeline.create(
  task="yolo",
  model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  class_names={"0":"person", "1":"bicycle", "2":"car", "3":"motorcycle"}

)
images = ["basilica.jpg"]

# run inference on image file
pipeline_outputs = yolo_pipeline(images=images)
print(pipeline_outputs.labels)
# [['motorcycle', 'car', 'person', 'car', 'car', 'person', 'person', 'person', 'motorcycle', 'person', 'person', 'person', 'person', 'person']]
```

#### IOU and Conf Threshold
We can also configure the thresholds for making detections in YOLO. 

```python
from deepsparse import Pipeline

# download onnx from sparsezoo and compile with batch size 1
sparsezoo_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none"
yolo_pipeline = Pipeline.create(
  task="yolo",
  model_path=sparsezoo_stub
)

images = ["basilica.jpg"]

# low threshold inference
pipeline_outputs_low_conf = yolo_pipeline(images=images, iou_thres=0.3, conf_thres=0.1)
print(len(pipeline_outputs_low_conf.boxes[0]))
# 37 <<makes 37 predictions>>

# high threshold inference
pipeline_outputs_high_conf = yolo_pipeline(images=images, iou_thres=0.5, conf_thres=0.8)
print(len(pipeline_outputs_high_conf.boxes[0]))
# 1 <<makes 1 prediction>>
```

### Cross Use Case Functionality

Check out the [Pipeline User Guide](../../user-guide/deepsparse-pipelines.md) for more details on configuring a Pipeline.

## DeepSparse Server

Built on the popular FastAPI and Uvicorn stack, DeepSparse Server enables you to set up a REST endpoint for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all the utilities provided by Pipelines.

Spin up the server:
```bash
deepsparse.server \
  --task yolo \
  --model_path zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none
```

Making a request.
```python
import requests
import json

url = 'http://0.0.0.0:5543/predict/from_files'
path = ['basilica.jpg'] # list of images for inference
files = [('request', open(img, 'rb')) for img in path]
resp = requests.post(url=url, files=files)
annotations = json.loads(resp.text) # dictionary of annotation results
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
print(labels)

# [['3.0', '2.0', '2.0', '0.0', '0.0', '2.0', '0.0', '0.0', '0.0', '3.0', '0.0', '0.0', '0.0', '0.0', '3.0', '9.0', '0.0', '2.0', '0.0', '0.0']]
```

#### Use Case Specific Arguments

To use the `image_size` or `class_names` argument, create a Server configuration file for passing the arguments via kwargs.

This configuration file sets `class_names` to `coco`:

```yaml
# yolo-config.yaml
endpoints:
  - task: yolo
    model: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none
    kwargs:
      class_names:
        '0': person
        '1': bicycle
        '2': car
        '3': motorcycle
      image_size: 320
```

Start the server: 
```bash 
deepsparse.server --config-file yolo-config.yaml
```

Making a request:
```python
import requests, json

url = 'http://0.0.0.0:5543/predict/from_files'
path = ['basilica.jpg'] # list of images for inference
files = [('request', open(img, 'rb')) for img in path]
resp = requests.post(url=url, files=files)
annotations = json.loads(resp.text)
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
print(labels)
# [['person', 'person', 'car', 'person', 'motorcycle', 'person', 'person', 'person', 'motorcycle', 'person']]
```

### Cross Use Case Functionality

Check out the [Server User Guide](../../user-guide/deepsparse-server.md) for more details on configuring a Server.
