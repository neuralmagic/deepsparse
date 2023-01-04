# DeepSparse

This guide explains how to deploy YOLOv5 with Neural Magic's DeepSparse.

## About DeepSparse

Welcome to software-delivered AI.

DeepSparse is an inference runtime offering GPU-class performance on CPUs. For the first time, your deep learning workloads can meet the performance 
demands of production without the complexity and costs of hardware accelerators.

Simply put, DeepSparse gives you the performance of GPUs and the simplicity of software:
- **Flexible Deployments**: Run consistently across cloud, data center, and edge with any hardware provider from Intel to AMD to ARM
- **Near-Infinite Scalability**: Scale vertically from 1 to 192 cores, out with standard Kubernetes, or fully-abstracted with Serverless
- **Easy Integration**: Clean APIs for integrating your model into an application and monitoring it in production

**[Start your 90 day Free Trial](https://neuralmagic.com/deepsparse-free-trial/?utm_campaign=free_trial&utm_source=ultralytics_github).**

### How Does DeepSparse Achieve GPU-Class Performance?

DeepSparse takes advantage of model sparsity to gain its performance speedup. 

Sparsification through pruning and quantization is a broadly studied technique, allowing reductions of 10x in the size and compute needed to 
execute a network, while maintaining high accuracy. DeepSparse is sparsity-aware, so it skips the multiply-adds by 0, shrinking amount of compute
in a forward pass. Since the sparse computation is memory bound, DeepSparse executes the network depth-wise, breaking the problem into Tensor Columns, 
vertical stripes of computation that fit in cache.

<p align="center">
  <img width="60%" src="sparse-network.svg">
</p>

Sparse computation, executed depth-wise in cache, allows DeepSparse to deliver GPU-class performance on CPUs!

### How Do I Create A Sparse Version of YOLOv5 Trained on My Data?

Neural Magic's open-source model repository SparseZoo and open-source model optimization library SparseML are integrated with Ultralytics YOLOv5, making 
it easy to fine-tune a pre-sparsified version of any YOLOv5 model onto custom data with a single CLI command.

[Checkout Neural Magic's YOLOv5 documentation for more details](https://docs.neuralmagic.com/use-cases/object-detection/sparsifying).

## DeepSparse Usage

We will walk through an example benchmarking and deploying a sparse version of YOLOv5s with DeepSparse.

### Install DeepSparse

Run the following to install DeepSparse. We recommend you use a virtual enviornment.

```bash
pip install deepsparse[server,yolo,onnxruntime]
```

### Collect an ONNX File

DeepSparse accepts a model in the ONNX format, passed either as:
- A SparseZoo stub which identifies an ONNX file in the SparseZoo
- A Local Path to an ONNX model in a filesystem

We will compare the pruned-quantized YOLOv5s to the standard dense YOLOv5s, identified by the following SparseZoo stubs:
```
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
```

### Benchmark Performance

DeepSparse includes a benchmarking script to test performance. The examples below use an AWS `c6i.4xlarge` instance (8 cores).

#### ONNX Runtime Baseline

The performance benchmarking script includes an option to run with ONNX Runtime as a baseline. Run the following:

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -nstreams 1 -e onnxruntime

> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
> Batch Size: 1
> Scenario: sync
> Throughput (items/sec): 34.8857
> Latency Mean (ms/batch): 28.6573
> Latency Median (ms/batch): 28.8170
> Latency Std (ms/batch): 0.6401
> Iterations: 349
```
At batch 1, ORT achieves 35 images/sec with dense YOLOv5s.

#### DeepSparse Dense Performance

While DeepSparse gets its best performance with sparse models, it also runs dense models well. Run the following:

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -nstreams 1

> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
> Batch Size: 1
> Scenario: sync
> Throughput (items/sec): 31.7332
> Latency Mean (ms/batch): 31.4978
> Latency Median (ms/batch): 31.4631
> Latency Std (ms/batch): 0.4734
> Iterations: 318
```

At batch 1, DeepSparse achieves 32 images/sec with dense YOLOv5s, a **XXX**!

#### DeepSparse Sparse Performance

When sparsity is applied to the model, DeepSparse's performance is even better. Run the following:

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none -s sync -b 1 -nstreams 1

Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
Batch Size: 1
Scenario: sync
Throughput (items/sec): 74.5251
Latency Mean (ms/batch): 13.4018
Latency Median (ms/batch): 13.4672
Latency Std (ms/batch): 0.2401
Iterations: 746
```

At batch 1, DeepSparse achieves 75 images/sec with a pruned-quantized YOLOv5s, a **2.1x performance improvement over ORT**!

#### Batch 64 Performance Comparison

In latency-insensitive scenarios with large batch sizes, DeepSparse's performance relative to ORT is even stronger.

ORT achieves 30 images/sec at batch 64:
```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 64 -nstreams 1 -e onnxruntime

Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
Batch Size: 64
Scenario: sync
Throughput (items/sec): 29.8662
Latency Mean (ms/batch): 2141.7437
Latency Median (ms/batch): 2116.9475
Latency Std (ms/batch): 58.9171
Iterations: 5
```

DeepSparse achieves 124 images/sec at batch 64, a **4.1x performance improvement over ORT**!

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none -s sync -b 64 -nstreams 1

> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 123.9485
> Latency Mean (ms/batch): 516.3261
> Latency Median (ms/batch): 516.0349
> Latency Std (ms/batch): 1.2869
> Iterations: 20
```

### Deploy a Model

DeepSparse offers convenient APIs for integrating your model into an application.  

To try the deployment examples below, pull down a sample image for the example and save as `basilica.jpg` with the following command:
```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

#### Python API
  
`Pipelines` wrap pre-processing and output post-processing around the runtime, providing a clean inferface for adding DeepSparse to an application. 
The DeepSparse-Ultralytics integration includes an out-of-the-box `Pipeline` that accepts raw images and outputs the bounding boxes.

Create a `Pipeline` and run inference:

```python
from deepsparse import Pipeline

# list of images in local filesystem
images = ["basilica.jpg"]

# create Pipeline
model_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none"
yolo_pipeline = Pipeline.create(
    task="yolo",
    model_path=model_stub,
)

# run inference on images, recieve bounding boxes + classes
pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
print(pipeline_outputs)
```

#### HTTP Server
  
DeepSparse Server runs on top of the popular FastAPI web framework and Uvicorn web server. With just a single CLI command, you can easily setup a model 
service endpoint with DeepSparse. The Server supports any Pipeline from DeepSparse, including object detection with YOLOv5, enabling you to send raw 
images to the endpoint and recieve the bounding boxes.

Spin up the Server with the pruned-quantized YOLOv5s:

```bash
deepsparse.server \
    --task yolo \
    --model_path zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65_quant-none
```

An example request, using Python's `requests` package:
```python
import requests, json

# list of images for inference (local files on client side)
path = ['basilica.jpg'] 
files = [('request', open(img, 'rb')) for img in path]

# send request over HTTP to /predict/from_files endpoint
url = 'http://0.0.0.0:5543/predict/from_files'
resp = requests.post(url=url, files=files)

# response is returned in JSON
annotations = json.loads(resp.text) # dictionary of annotation results
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
```

## Get Started With DeepSparse

**Research or Testing?** DeepSparse Community is free for research and testing. Production deployments require DeepSparse Enterprise.

**Want to Try DeepSparse Enterprise?** [Start your 90 day free trial](https://neuralmagic.com/deepsparse-free-trial/?utm_campaign=free_trial&utm_source=ultralytics_github).
