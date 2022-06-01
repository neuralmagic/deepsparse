# YOLOv5 Inference Pipelines


YOLOv5 integration allows serving and benchmarking sparsified [Ultralytics yolo](https://github.com/ultralytics/yolo) models.  
This integration allows for leveraging the DeepSparse Engine to run YOLOv5 inference with GPU-class performance directly on the CPU.

The DeepSparse Engine is taking advantage of sparsity within neural networks to 
reduce compute required as well as accelerate memory-bound workloads. The Engine is particularly effective when leveraging sparsification
methods such as [pruning](https://neuralmagic.com/blog/pruning-overview/) and [quantization](https://arxiv.org/abs/1609.07061). 
These techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics. 

This integration currently supports the original YOLOv5 the updated V6.1 architectures

## Getting Started


Before you start your adventure with the DeepSparse Engine, make sure that your machine is 
compatible with our [hardware requirements](https://docs.neuralmagic.com/deepsparse/source/hardware.html).

### Installation

```pip install deepsparse```

### Model Format
By default, to deploy YOLOv5 using DeepSparse Engine it is required to supply the model in the ONNX format. 
This grants the Engine the flexibility to serve any model in a framework-agnostic environment. 

Below we describe two possibilities to obtain the required ONNX model.

### Exporting the onnx file from the contents of a local directory
This pathway is relevant if you intend to deploy a model created using [SparseML](https://github.com/neuralmagic/sparseml) library. 
For more information refer to the appropriate YOLOv5 integration documentation in SparseML.
1. After training your model with `SparseML`, locate the `.pt` file for the model you'd like to export
2. Run the `SparseML` integrated YOLOv5 onnx export script the CLI command as below
```bash
sparseml.yolov5.export_onnx --weights path/to/your/model --dynamic
```
This will create an `.onnx` file in the same directory and with the same root name as the 
model weights file. 

###  Directly using the SparseZoo stub
Alternatively, you can skip the onnx model export process by downloading all the required model data directly from Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com/).
Example:
```python
from sparsezoo import Zoo

# you can lookup an appropriate model stub here: https://sparsezoo.neuralmagic.com/
model_stub = "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98"
# directly download the model data to your local directory
model = Zoo.download_model_from_stub(model_stub)

# the onnx model file is there, ready for deployment
import os 
os.path.isfile(os.path.join(model.dir_path, "model.onnx"))
>>True
```


## Deployment

### Python API
Python API is the default interface for running the inference with the DeepSparse Engine. We can use it to run inference on local images. If you don't have an image ready, pull a sample image down with

```
wget -O abbey_road.jpg  https://upload.wikimedia.org/wikipedia/en/4/42/Beatles_-_Abbey_Road.jpg
```

[List of the YOLOv5 SparseZoo Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=question_answering)

```python
from deepsparse.pipeline import Pipeline

model_stub = "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98"
images = ["abbey_road.jpg"]

od_pipeline = Pipeline.create(
    task="yolo",
    model_path=model_stub,
)

pipeline_outputs = od_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
print(pipeline_outputs)
```

### Annotate CLI
You can also annotate an image source directly with the following CLI command
```bash
deepsparse.object_detection.annotate --source abbey_road.jpg #Try --source 0 to annotate your live webcam feed
```

### DeepSparse Server
As an alternative to Python API, the DeepSparse inference server allows you to serve ONNX models and pipelines in HTTP.

#### Spinning Up with DeepSparse Server
Install the server:
```bash
pip install deepsparse[server]
```

Run `deepsparse.server --help` to look up the CLI arguments:
```bash
  Start a DeepSparse inference server for serving the models and pipelines
  given within the config_file or a single model defined by task, model_path,
  and batch_size

  Example config.yaml for serving:

  models:
      - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
        batch_size: 1
        alias: question_answering/dense
      - task: question_answering
        ...

Options:
  --host TEXT                     Bind socket to this host. Use --host 0.0.0.0
                                  to make the application available on your
                                  local network. IPv6 addresses are supported,
                                  for example: --host '::'. Defaults to
                                  0.0.0.0
  --port INTEGER                  Bind to a socket with this port. Defaults to
                                  5543.
  --workers INTEGER               Use multiple worker processes. Defaults to
                                  1.
  --log_level [debug|info|warn|critical|fatal]
                                  Sets the logging level. Defaults to info.
  --config_file TEXT              Configuration file containing info on how to
                                  serve the desired models.
  ...
```


Example CLI Command to spin up the server:

```bash
deepsparse.server \
    --task yolo \
    --model_path zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
```

Sample request to the server:

```python
import requests
url = "http://localhost:5543/predict" # Server's port default to 5543
obj = {}
response = requests.post(url, json=obj)
response.text
```

### Benchmarking
The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs. Want to find out how fast our sparse YOLOv5 ONNX models perform inference? 
You can quickly do benchmarking tests on your own with a single CLI command!

You only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94

>> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94
>> Batch Size: 1
>> Scenario: async
>> Throughput (items/sec): 120.5551
>> Latency Mean (ms/batch): 99.3043
>> Latency Median (ms/batch): 99.1637
>> Latency Std (ms/batch): 2.7053
>> Iterations: 1210
```

To learn more about benchmarking, refer to the appropriate documentation.
Also, check out our [Benchmarking tutorial](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)!

## Tutorials:
For a deeper dive into using transformers within the Neural Magic ecosystem, refer to the detailed tutorials on our [website](https://neuralmagic.com/use-cases/#computervision).

## Support
For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues).