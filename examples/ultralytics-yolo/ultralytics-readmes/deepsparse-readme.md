# DeepSparse

This guide explains how to deploy YOLOv5 with Neural Magic's DeepSparse to achieve GPU-class performance on commodity CPUs.

## About DeepSparse

Welcome to software-delivered AI.

DeepSparse is an inference runtime offering GPU-class performance on CPUs. For the first time, your deep learning workloads can meet the performance 
demands of production without the complexity and costs of hardware accelerators.

Simply put, DeepSparse gives you the performance of GPUs and the simplicity of software:
- **Exceptional Performance**: Deploy state-of-the-art models with GPU-class performance on low-cost commodity CPUs
- **Flexible Deployment**: Run consistently across cloud, data center, and edge with any hardware provider from Intel to AMD to ARM
- **Near-Infinite Scalability**: Scale vertically from 1 to 192 cores, out with standard Kubernetes, or fully-abstracted with Serverless
- **Ease of Integration**: Clean APIs for integrating your model into an application and monitoring it in production

### How Does DeepSparse Achieve GPU-Class Performance?

DeepSparse takes advantage of model sparsity to gain its performance speedup. 

Sparsification through pruning and quantization is a broadly studied technique, allowing reductions of 10x in the size and compute needed to 
execute a network, while maintaining high accuracy. DeepSparse is sparsity-aware, so it skips the multiply-adds by 0, shrinking the number of instructions 
in a forward pass. Since the sparse computation is memory bound, DeepSparse executes the network depth-wise, breaking the problem into Tensor Columns, 
vertical stripes of computation that fit in cache without having to read or write to memory.

<p align="center">
  <img width="75%" src="sparse-network.svg">
</p>

Sparse computation, executed depth-wise in cache, allows DeepSparse to deliver GPU-class performance on CPUs!

### How Do I Create A Sparse Version of YOLOv5 To Run With DeepSparse?

Neural Magic has open-source sparse versions of each YOLOv5 model, available for use from the [SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1).

Neural Magic's SparseML library is integrated with Ultralytics, enabling you to create a sparse model trained on your data. SparseML
allows you to transfer learn from pre-sparsified YOLOv5 models in SparseZoo and to apply pruning and quantization to your YOLOv5 model from scratch. See [our YOLOv5 documentation](https://docs.neuralmagic.com/use-cases/object-detection/sparsifying) for more details.

## DeepSparse Usage

We will walk through an example benchmarking and deploying a sparse version of YOLOv5s with DeepSparse.

Pull down a sample image for the example and save as `basilica.jpg` with the following command:
```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

### Install DeepSparse

Run the following to install DeepSparse. We recommend you use a virtual enviornment.

```bash
pip install deepsparse[server,yolo]
```

### Collect an ONNX File

DeepSparse accepts a model in the ONNX format. It can be one of two options:   
- A SparseZoo Stub which identifies a pre-sparsified model in the SparseZoo and downloads the ONNX file for you
- A Local Path to an ONNX model in a filesystem. The [SparseML YOLOv5 docs](https://docs.neuralmagic.com/use-cases/object-detection/sparsifying) include 
an example of how to export a model to ONNX.

We will compare the pruned-quantized YOLOv5s from the SparseZoo to the standard dense YOLOv5s, identified by the following stubs:
```
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
```

### Benchmark Performance

We will first demonstrate DeepSparse's performance gains. 

DeepSparse includes a benchmarking script to test performance. We look at the realtime batch 1 scenario using an AWS `c6i.4xlarge` instance (8 cores).

#### ONNX Runtime Dense Performance

As a baseline, let's look at ONNX Runtime's performance. Install ONNXRuntime with `pip install onnxruntime`.

We can see ONNX Runtime achieves 17 images/sec at batch 1 with dense YOLOv5-s:
```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -e onnxruntime

>> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 17.2266
>> Latency Mean (ms/batch): 58.0394
>> Latency Median (ms/batch): 58.0130
>> Latency Std (ms/batch): 0.2080
>> Iterations: 173
```

#### DeepSparse Dense Performance

While DeepSparse gets the best performance with sparse models, it also has strong performance for standard dense models.

We can see DeepSparse achieves 35 images/sec at batch 1 with dense YOLOv5-s, a **2x** performance improvement over ORT:

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1

>> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 35.2821
>> Latency Mean (ms/batch): 28.3274
>> Latency Median (ms/batch): 28.3064
>> Latency Std (ms/batch): 0.1022
>> Iterations: 353
```

#### DeepSparse Sparse Performance

With the sparse version of YOLOv5-s, DeepSparse's performance is even stronger.

We can see DeepSparse achieves 80 images/sec with sparse YOLOv5-s, a **4.7x** performance improvement over ORT.

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 -s sync -b 1

>> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 79.8339
>> Latency Mean (ms/batch): 12.5111
>> Latency Median (ms/batch): 12.5534
>> Latency Std (ms/batch): 0.1546
>> Iterations: 799
```

#### Batch 64 Performance Comparison

In latency-insensitive scenarios (where batch sizes are large), DeepSparse's performance gain relative to ONNX Runtime is even stronger.

ORT achieves 15 images/sec at batch 64:
```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -e onnxruntime

>> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 14.7454
>> Latency Mean (ms/batch): 4339.7396
>> Latency Median (ms/batch): 4190.1105
>> Latency Std (ms/batch): 211.6520
>> Iterations: 3
```

DeepSparse achieves 148 images/sec at batch 64, a **10x** performance improvement over ONNX Runtime:

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 -s sync -b 64

>> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 147.6844
>> Latency Mean (ms/batch): 433.3391
>> Latency Median (ms/batch): 433.1426
>> Latency Std (ms/batch): 0.9764
>> Iterations: 24
```

### Deploy a Model

Beyond offering exceptional performance, DeepSparse offers convenient APIs for integrating your model into an application. There are two
primary APIs for deploying YOLOv5 on DeepSparse.

#### Python API: run inference on the client side or within an application
  
`Pipelines` wrap image pre-processing and output post-processing around the runtime. The DeepSparse-Ultralytics integration includes an 
out-of-the-box `Pipeline` that accepts raw images and outputs the bounding boxes.

Create a `Pipeline` for inference:

```python
from deepsparse import Pipeline

# list of images in local filesystem
images = ["basilica.jpg"]

# create Pipeline
model_stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94"
yolo_pipeline = Pipeline.create(
    task="yolo",
    model_path=model_stub,
)

# run inference on images, recieve bounding boxes + classes
pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
print(pipeline_outputs)
```

#### HTTP Server: easily setup a model service behind a REST API
  
The DeepSparse Server runs on top of the popular FastAPI web framework and Uvicorn web server such that you can query a model via HTTP. 
The server supports any task from DeepSparse, including object detection.

Spin up the server with sparse by running the following from the command line: 

```bash
deepsparse.server \
    yolo \
    --model_path zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
```

An example request, using Python's `requests` package:
```python
import requests
import json

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

**Want to Try DeepSparse Enterprise?** Neural Magic has a [60 day free trail](https://neuralmagic.com/deepsparse-free-trial/?utm_campaign=free_trial&utm_source=ultralytics_github).
