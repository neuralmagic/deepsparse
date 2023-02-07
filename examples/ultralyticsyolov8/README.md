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

# Deploy YOLOv8 with DeepSparse

For production deployments in real-world applications, inference speed is crucial in determining the overall cost and responsiveness of the system. DeepSparse is an inference runtime focused on making deep learning models like YOLOv8 run fast on CPUs. While DeepSparse achieves its best performance with inference-optimized sparse models, it can also run standard, off-the-shelf models efficiently.

Let’s export the standard YOLOv8 model to ONNX and run some benchmarks on an AWS c6i.16xlarge instance. 

```bash
# Install packages for DeepSparse and YOLOv8
pip install deepsparse[yolo] ultralytics

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

DeepSparse also offers some convenient utilities for integrating a model into your application. For instance, you can annotate images or video using YOLOv8:

```bash
# Download example code, model, and sample image
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/ultralyticsyolov8
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg

# Annotate image with DeepSparse
python annotate.py --source basilica.jpg  --model_filepath yolov8n.onnx
```

<img width="1041" alt="Screenshot 2023-01-11 at 6 53 46 PM" src="https://user-images.githubusercontent.com/3195154/211942937-1d32193a-6dda-473d-a7ad-e2162bbb42e9.png">


DeepSparse’s performance can be pushed even further by optimizing the model for inference. DeepSparse is built to take advantage of models that have been optimized with weight pruning and quantization—techniques that dramatically shrink the required compute without dropping accuracy. Through our One-Shot optimization methods, which will be made available in an upcoming product called Sparsify, we have produced YOLOv8s and YOLOv8n ONNX models that have been quantized to INT8 while maintaining at least 99% of the original FP32 mAP@0.5. This was achieved with just 1024 samples and no back-propagation. You can download the quantized models [here](https://drive.google.com/drive/folders/1vf4Es-8bxhx348TzzfhvljMQUo62XhQ4?usp=sharing). 

Run the following to benchmark performance:

```bash
deepsparse.benchmark yolov8n_quant.onnx --scenario=sync --input_shapes="[1,3,640,640]"
> Throughput (items/sec): 525.0226
> Latency Mean (ms/batch): 1.9047

deepsparse.benchmark yolov8s_quant.onnx --scenario=sync --input_shapes="[1,3,640,640]"
> Throughput (items/sec): 209.9472
> Latency Mean (ms/batch): 4.7631
```

DeepSparse places commodity CPUs right next to the A100 GPU, which achieves [~1ms latency](https://github.com/ultralytics/ultralytics#models). Check out our performance benchmarks for YOLOv8 on Amazon EC2 C6i Instances. DeepSparse is 4X faster at FP32 and 10X faster at INT8 than all other CPU alternatives.  
