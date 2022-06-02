# Image Classification Inference Pipelines


[DeepSparse] Image Classification integration allows serving and benchmarking 
sparsified image classification models. This integration enables leveraging 
the [DeepSparse] Engine to run inference with GPU-class performance directly on 
the CPU.

The [DeepSparse] Engine takes advantage of sparsity within neural networks to 
reduce compute as well as accelerate memory-bound workloads. 
The Engine is particularly effective when leveraging sparsification methods 
such as [pruning](https://neuralmagic.com/blog/pruning-overview/) and 
[quantization](https://arxiv.org/abs/1609.07061). These techniques result in 
significantly more performant and smaller models with limited to no effect on 
the baseline metrics.

## Getting Started

Before you start your adventure with the [DeepSparse] Engine, make sure that 
your machine is compatible with our [hardware requirements].

### Installation

```pip install deepsparse```

### Model Format

By default, to deploy image classification models using the [DeepSparse] Engine,
the model should be supplied in the [ONNX] format. 
This grants the Engine the flexibility to serve any model in a framework-agnostic
manner. 

Below we describe two possibilities to obtain the required ONNX model.

#### Exporting the onnx file from the contents of a local checkpoint

This pathway is relevant if you intend to deploy a model created using [SparseML] library. 
For more information refer to the appropriate integration documentation in [SparseML].

1. The output of the `[SparseML]` training is saved to output directory `/{save_dir}` (e.g. `/trained_model`)
2. Depending on the chosen framework, the model files are saved to `model_path`=`/{save_dir}/{framework_name}/{model_tag}` (e.g `/trained_model/pytorch/resnet50/`)
3. To generate an onnx model, refer to the [script for image classification ONNX export](https://github.com/neuralmagic/sparseml/blob/main/src/sparseml/pytorch/image_classification/export.py).

Example:
```bash
sparseml.image_classification.export_onnx \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 \
    --checkpoint-path ~/checkpoints/resnet50_checkpoint.pth
```
This creates `model.onnx` file, in the parent directory of your `model_path`

####  Directly using the SparseZoo stub

Alternatively, you can skip the process of onnx model export by downloading all the required model data directly from Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com/).
Example:
```pycon
from sparsezoo import Zoo

# you can lookup an appropriate model stub here: https://sparsezoo.neuralmagic.com/
model_stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none"

# directly download the model data to your local directory
model = Zoo.download_model_from_stub(model_stub)

# the onnx model file is there, ready for deployment
import os 
os.path.isfile(os.path.join(model.dir_path, "model.onnx"))
>>>True
```


## Deployment

### Python API
Python API is the default interface for running the inference with the DeepSparse Engine.
The [SparseML] installation provides a CLI for sparsifying models for a specific task;
To find out more on how to sparsify Image Classification models refer to 
[SparseML Image Classification Documentation]

To learn about sparsification in more detail, refer to [SparseML docs](https://docs.neuralmagic.com/sparseml/)

#### Image Classification Pipeline

[List of Image Classification SparseZoo Models](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=classification&page=1)

```python
from deepsparse import pipeline
cv_pipeline = pipeline(
  task='image_classification', 
  model_path='zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none',  # Path to checkpoint or SparseZoo stub
)
input_image = ... # Read input images
inference = cv_pipeline(images=input_image)
```

### DeepSparse Server
As an alternative to Python API, the DeepSparse inference server allows you to serve ONNX models and pipelines in HTTP.
To learn more about the DeeepSparse server, refer to the [appropriate documentation](https://github.com/neuralmagic/deepsparse/tree/main/examples/classification).

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
      - task: image_classification
        model_path: zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none
        batch_size: 1
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

Example CLI Command to spin up the server with a 95% pruned `resnet50`:
```bash
deepsparse.server \
    --task image_classification \
    --model_path "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none" \
    --port 5543
```

Sample client for sending requests to the server:
`imagenet_client.py`
```python
from glob import glob
import pathlib
import click
import requests
import cv2
default_url = "http://localhost:5543/predict"
@click.command()
@click.option(
    '--url',
    default=default_url,
    help='The URL to the server',
    show_default=True,
)
@click.option(
    '--data',
    '--data',
    default="data",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
)
@click.option(
    '--max_samples',
    '--max-samples',
    default=100,
    type=int,
    help='The maximum number of samples to test',
)
def main(url, data, max_samples):
    input_path = pathlib.Path(data)
    input_file_names = (
        glob(f"{data}/*.jpg") + glob(f"{data}/*.jpeg") + glob(f"{data}/*.JPEG")
        if input_path.is_dir()
        else [data]
    )
    for index, image_path in enumerate(input_file_names):
        if index >= max_samples:
            break
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        payload = {
            "images": [image.tolist()]
        }
        response = requests.post(url, json=payload)
        print(response.json())
if __name__ == '__main__':
    main()
```

Invoke the client for 10 samples as follows:
```bash
python imagenet_client.py --data PATH/TO/IMAGE/DIRECTORY \
    --max-samples 10
```

### Benchmarking

The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs. 
Want to find out how fast our sparse ONNX models perform inference? 
You can quickly do benchmarking tests on your own with a single CLI command!

You only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:
```bash
deepsparse.benchmark zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none
```
Output:
```bash
Original Model Path: zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none
Batch Size: 1
Scenario: async
Throughput (items/sec): 299.2372
Latency Mean (ms/batch): 16.6677
Latency Median (ms/batch): 16.6748
Latency Std (ms/batch): 0.1728
Iterations: 2995
```

To learn more about benchmarking, refer to the appropriate documentation.
Also, check out our [Benchmarking tutorial](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)!

## Tutorials:
For a deeper dive into using image classification models within the Neural Magic
ecosystem, refer to the detailed tutorials on our [website](https://neuralmagic.com/):
- [CV Use Cases](https://neuralmagic.com/use-cases/#computervision)

## Support
For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues).


[DeepSparse]: https://github.com/neuralmagic/deepsparse
[hardware requirements]: https://docs.neuralmagic.com/deepsparse/source/hardware.html
[ONNX]: https://onnx.ai/
[SparseML]: https://github.com/neuralmagic/sparseml
[SparseML Image Classification Documentation]: https://github.com/neuralmagic/sparseml/tree/main/src/sparseml/pytorch/image_classification/README_image_classification.md