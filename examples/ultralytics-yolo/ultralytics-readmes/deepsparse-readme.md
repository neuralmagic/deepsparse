# DeepSparse: Deploy YOLOv5 with Realtime Latency on CPUs

Learn how to deploy YOLOv5 with Neural Magic's DeepSparse. 

## DeepSparse Overview

DeepSparse is an inference runtime offering GPU-class performance on CPUs and tooling to integrate ML into your application. 

CPU-only deployments take advantage of the low cost, flexibility, and scalability of software-delivered inference:
- Lower latency than a GPU, at lower cost
- Deploy the same model and runtime on any hardware from Intel to AMD to ARM and from cloud to data center to edge, including on pre-existing systems
- Scale vertically from 1 to 192 cores, tailoring the footprint to an app's exact needs
- Scale horizontally with standard Kubernetes, including using services like EKS/GKE
- Scale abstractly with serverless instances like GCP Cloud Run and AWS Lambda
- Integrate easily into "Deploy with code" provisioning systems
- No wrestling with drivers, operator support, and compatibility issues

With DeepSparse, you no longer need to pick between the performance of GPUs and the simplicity of software!

## How Does DeepSparse Achieve GPU-Class Performance with just CPUs?

DeepSparse uses sparsity in neural networks to gain its performance speedup.

When we say sparsity, we are talking about sparsity in the weights of the network. Sparsification through pruning and quantization is a broadly 
studied ML technique, allowing reductions of 10x in the size and theoretical compute needed to execute a neural network, without losing much accuracy.

There are two primary techniques for creating sparse models:
- **Pruning** removes redundant weights from a neural network. Because neural networks are highly overparameterized,
removing redundant weights has very little if any impact on the model's accuracy, especially when performed in a training-aware manner where
the non-zero weights can adjust to the new optimization space. By removing the useless weights and and settings them to 0, we can reduce the 
amount of computation needed to execute a forward pass.

- **Quantization** reduces the precision of the weights typically from FP32 to INT8. This reduces the amount of 
memory needed to represent a model. Just like pruning, quantization has very little imact on model accuracy, especially
when performed in a training-aware manner. By reducing the precision of the weights, the model can be executed more quickly as more
data can fit into the caches inside a CPU, which is often the bottleneck of the computation.

DeepSparse is uniquely architected to take advantage of sparsity to gain performance speedups. DeepSparse has implemented sparse 
versions of common operations in deep neural networks such as the convolution, and is able to effectively "skip" the multiply-adds by zero, 
dramatically shrinking the amount of computation executed in the forward pass. The deeply sparsified computation is memory bound, 
so DeepSparse execute the neural network depth-wise rather than layer-after-layer. It might seem like magic, but DeepSparse is able 
to break the network into Tensor Columns, vertical stripes of computation that fit completely in cache without having to read or write to memory.

Sparse computation, executed depth-wise in cache, allows us to deliver GPU-class performance on CPUs!

#### How Do I Create A Sparse Version of YOLOv5?

Neural Magic has created open-source sparsified versions of each YOLOv5 model, available for use from the [SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1).

Additionally, Neural Magic's SparseML library is integrated with Ultralytics, enabling you to create a sparse model trained on your data. These pathways
allow you to transfer learn from pre-sparsified YOLOv5 models from SparseZoo or apply pruning and quantization to your YOLOv5 model from scratch. See [our YOLOv5 documentation](https://docs.neuralmagic.com/use-cases/object-detection/sparsifying) for more details.

## Usage Example

We will walk through an example deploying a sparse version of YOLOv5s with DeepSparse, following these steps:
- Install DeepSparse
- Collect ONNX File
- Benchmark Latency/Throughput
- Deploy a Model

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

DeepSparse accepts a model in the ONNX format.

The `model_path` argument in the commands below tells DeepSparse where the ONNX file is. It can be one of two options:   
- `sparsezoo_stub` which identifies a pre-sparsified model in the SparseZoo and downloads the ONNX file for you
- `local_path` to an ONNX model `model_name.onnx` in a filesystem. The [SparseML YOLOv5 documentation](https://docs.neuralmagic.com/use-cases/object-detection/sparsifying)
includes an example of how to export a model to ONNX.

In the example below, we will compare the pruned-quantized YOLOv5s from the SparseZoo to the dense YOLOv5s, identified by the following stubs:
```
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
```

### Benchmark Performance

Let's demonstrate DeepSparse's performance gains.

DeepSparse includes a benchmarking script to test performance under a variety of scenarios. We will look at performance for batch-size 1 with 640x640 images, 
using an AWS `c6i.4xlarge` instance (8 cores).

#### ONNX Runtime Dense Performance (Batch 1)

As a baseline, let's look at ONNX Runtime's performance.

Install ONNXRuntime: 

```
pip install onnxruntime
```

Run the following to test ORTs's latency and throughput on dense YOLOv5s with batch 1:

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -e onnxruntime
```

The output is:
```
Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
Batch Size: 1
Scenario: sync
Throughput (items/sec): 17.2266
Latency Mean (ms/batch): 58.0394
Latency Median (ms/batch): 58.0130
Latency Std (ms/batch): 0.2080
Iterations: 173
```

#### DeepSparse Dense Performance (Batch 1)

While DeepSparse has the best performance with sparse models, it also has a speedup relative to ONNX Runtime on dense models.

Run the following to test DeepSparse's latency and throughput on dense YOLOv5s with batch 1:

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1
```

The output is:
```
Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
Batch Size: 1
Scenario: sync
Throughput (items/sec): 35.2821
Latency Mean (ms/batch): 28.3274
Latency Median (ms/batch): 28.3064
Latency Std (ms/batch): 0.1022
Iterations: 353
```

DeepSparse offers a **2x** performance improvement over ONNX Runtime as throughput increased from 17 to 35 images/sec, without even sparsifying the model!

#### DeepSparse Sparse Performance (Batch 1)

DeepSparse offers the best performance with sparse models.

Run the following to test DeepSparse's latency and throughput on pruned-quantized YOLOv5s with batch 1:

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 -s sync -b 1
```

The output is:
```
Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
Batch Size: 1
Scenario: sync
Throughput (items/sec): 79.8339
Latency Mean (ms/batch): 12.5111
Latency Median (ms/batch): 12.5534
Latency Std (ms/batch): 0.1546
Iterations: 799
```

By utilizing sparsity, DeepSparse offers a **4.7x** performance improvement over ONNX Runtime as throughput increased from 17 to 80 images/sec!

#### DeepSparse vs ONNX Runtime (Batch 64)

In latency-insensitive scenarios (where batch sizes are large), DeepSparse's performance gain relative to ONNX Runtime is even stronger.

**ONNX Runtime at Batch 64:**

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none -s sync -b 1 -e onnxruntime
```

The output is:
```
Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
Batch Size: 64
Scenario: sync
Throughput (items/sec): 14.7454
Latency Mean (ms/batch): 4339.7396
Latency Median (ms/batch): 4190.1105
Latency Std (ms/batch): 211.6520
Iterations: 3
```

**DeepSparse at Batch 64:**

```
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 -s sync -b 64
```

The output is:
```
Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
Batch Size: 64
Scenario: sync
Throughput (items/sec): 147.6844
Latency Mean (ms/batch): 433.3391
Latency Median (ms/batch): 433.1426
Latency Std (ms/batch): 0.9764
Iterations: 24
```

In the throuput scenario, DeepSparse offers a **10x** performance improvement over ONNX Runtime as throughput increased from 15 to 148 images/sec!

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
