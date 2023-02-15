
# YOLOv8 Inference Pipelines

DeepSparse allows accelerated inference, serving, and benchmarking of sparsified [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models.  

> "Ultralytics YOLOv8, developed by Ultralytics, is a cutting-edge, state-of-the-art (SOTA) model that builds upon 
> the success of previous YOLO versions and introduces new features and improvements to further boost performance 
> and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide 
> range of object detection, image segmentation, and image classification tasks."
This integration allows for leveraging the DeepSparse Engine to run the sparsified YOLOv8 inference with GPU-class performance directly on the CPU.

The DeepSparse Engine is taking advantage of sparsity within neural networks to 
reduce compute required as well as accelerate memory-bound workloads. The engine is particularly effective when leveraging sparsification
methods such as [pruning](https://neuralmagic.com/blog/pruning-overview/) and [quantization](https://arxiv.org/abs/1609.07061). 
These techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics.

## Getting Started

Before you start your adventure with the DeepSparse Engine, make sure that your machine is 
compatible with our [hardware requirements](https://docs.neuralmagic.com/deepsparse/source/hardware.html).

### Installation

```pip install deepsparse[yolov8]```
### Model Format
By default, to deploy YOLOv8 using DeepSparse Engine it is required to supply the model in the ONNX format. 
This grants the engine the flexibility to serve any model in a framework-agnostic environment. 
Below we describe three possibilities to obtain the required ONNX model.
### Fetching the original (non-compressed) YOLOv8 directly from the Ultralytics repository
```bash
# Install packages for DeepSparse and YOLOv8
pip install deepsparse[yolov8]
# Export YOLOv8n and YOLOv8s ONNX models
yolo task=detect mode=export model=yolov8n.pt format=onnx opset=13
```
This will save `yolov8n.onnx` to your disk.

### Exporting the ONNX File From the Contents of a Local Directory
This pathway is relevant if you intend to deploy a model created using the [SparseML](https://github.com/neuralmagic/sparseml) library. 
For more information refer to the [appropriate YOLOv8 integration documentation in SparseML](https://github.com/neuralmagic/sparseml/tree/main/src/sparseml/yolov8).
After training your model with `SparseML`, locate the `.pt` file for the model you'd like to export and run the `SparseML` integrated YOLOv8 ONNX export script below.

```bash
sparseml.yolov8.export_onnx \
    --weights path/to/your/model \
    --dynamic #Allows for dynamic input shape
```
This creates a `model.onnx` file, in the directory of your `weights` (e.g. `runs/train/weights/model.onnx`).

#### Fetching Sparsified YOLOv8 Models
DeepSparse’s performance can be pushed even further by optimizing the model for inference. DeepSparse is built to take advantage of models that have been optimized with weight pruning 
and quantization—techniques that dramatically shrink the required compute without dropping accuracy. Through our One-Shot optimization methods, which will be made available in an upcoming 
product called Sparsify, we have produced YOLOv8s and YOLOv8n ONNX models that have been quantized to INT8 while maintaining at least 99% of the original FP32 mAP@0.5. 
This was achieved with just 1024 samples and no back-propagation. You can download the quantized models [here](https://drive.google.com/drive/folders/1vf4Es-8bxhx348TzzfhvljMQUo62XhQ4?usp=sharing).

## Deployment Example
The following example uses pipelines to run a pruned and quantized YOLOv8 model for inference. As input, the pipeline ingests a list of images and returns for each image the detection boxes in numeric form.

If you don't have an image ready, pull a sample image down with

```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

```python
from deepsparse import Pipeline

model_path = "../../../yolov8n.onnx"  # or "yolov8n_quant.onnx"
images = ["basilica.jpg"]
yolo_pipeline = Pipeline.create(
    task="yolov8",
    model_path=model_path,
)
pipeline_outputs = yolo_pipeline(images=images)
```
<img width="1041" alt="Screenshot 2023-01-11 at 6 53 46 PM" src="https://user-images.githubusercontent.com/3195154/211942937-1d32193a-6dda-473d-a7ad-e2162bbb42e9.png">

#### Annotate CLI
You can also use the `annotate` command to have the engine save an annotated photo on the disk.
```bash
deepsparse.yolov8.annotate --source basilica.jpg --model_filepath "yolov8n.onnx" # or "yolov8n_quant.onnx"
```

Running the above command will create an `annotation-results` folder and save the annotated image inside.

#### Annotate CLI
You can also use the `eval` command to have the engine run inference on a dataset (in the same fashion as it is being done in `ultralytics` module)
```bash
deepsparse.yolov8.eval --model_path yolov8n.onnx
```
Output:
```bash
DeepSparse, Copyright 2021-present / Neuralmagic, Inc. version: 1.4.0.20230211 COMMUNITY | (bacdaef4) (release) (optimized) (system=avx2, binary=avx2)
Ultralytics YOLOv8.0.36 🚀 Python-3.8.10 torch-1.12.1+cu116 CUDA:0 (NVIDIA RTX A4000, 16117MiB)
val: Scanning /home/ubuntu/damian/deepsparse/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 8/8 [00:07<00:00,  1.04it/s]
                   all        128        929      0.651      0.532      0.606      0.453
                person        128        254      0.805      0.667      0.764      0.543
               bicycle        128          6      0.661      0.328      0.329      0.232
                   car        128         46      0.817      0.196      0.269      0.181
            motorcycle        128          5      0.602        0.8       0.88      0.684
...
```

Running the above command will create an `annotation-results` folder and save the annotated image inside.

### Benchmarking
The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs. Want to find out how fast our sparse YOLOv8 ONNX models perform inference? 
You can quickly do benchmarking tests on your own with a single CLI command!

You only need to provide the model path of the local ONNX model to get started:

```bash
# Install packages for DeepSparse and YOLOv8
pip install deepsparse[yolov8] ultralytics
# Export YOLOv8n and YOLOv8s ONNX models
yolo task=detect mode=export model=yolov8n.pt format=onnx opset=13
yolo task=detect mode=export model=yolov8s.pt format=onnx opset=13
# Benchmark with DeepSparse!
deepsparse.benchmark yolov8n.onnx --scenario=sync --input_shapes="[1,3,640,640]"
> Throughput (items/sec): 198.3282
> Latency Mean (ms/batch): 5.0366
deepsparse.benchmark yolov8s.onnx --scenario=sync --input_shapes="[1,3,640,640]"
> Throughput (items/sec): 68.3909
> Latency Mean (ms/batch): 14.6101
```

```bash
deepsparse.benchmark yolov8n_quant.onnx --scenario=sync --input_shapes="[1,3,640,640]"
> Throughput (items/sec): 525.0226
> Latency Mean (ms/batch): 1.9047
deepsparse.benchmark yolov8s_quant.onnx --scenario=sync --input_shapes="[1,3,640,640]"
> Throughput (items/sec): 209.9472
> Latency Mean (ms/batch): 4.7631
```

DeepSparse places commodity CPUs right next to the A100 GPU, which achieves [~1ms latency](https://github.com/ultralytics/ultralytics#models). Check out our performance benchmarks for YOLOv8 on Amazon EC2 C6i Instances. DeepSparse is 4X faster at FP32 and 10X faster at INT8 than all other CPU alternatives.  

To learn more about benchmarking, refer to the appropriate documentation.
Also, check out our [Benchmarking tutorial](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)!

## Tutorials:
For a deeper dive into using YOLOv8 within the Neural Magic ecosystem, refer to the detailed tutorials on our [website](https://neuralmagic.com/use-cases/#computervision).

## Support
For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues).
