# Wrapping a Model Deployment with Docker

This README outlines the many options you have for storing a deepsparse model inside
a docker container.

The [Dockerfile](Dockerfile) shows one case of this: running `sparsezoo.download`
when building the image, and passing to `deepsparse.server task`. Example usage:

```bash
docker build --build-arg TASK=qa --build-arg MODEL_STUB=zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none -t qa .
```

## What is a Model Deployment?

It is a directory with at least a `model.onnx` file in it, but usually other configuration files.

## How to create a Model Deployment?

There are many options!

#### Use sparsezoo.download

```bash
sparsezoo.download --save-dir <dir> <model stub>
```

The deployment directory will be under `<dir>/deployment`.

#### Use sparsezoo.Model in python

This enables you to only download the deployment directory:

```python
from sparsezoo import Model

stub = ...
save_dir = ...
model = Model(stub, download_path=save_dir)

# Downloads and prints the download path of the model
print(model.deployment.path)
```

## Storing a Model Deployment in a Dockerfile

There are two methods here:
1. Download the deployment directory to your host machine and copy it in
2. Use sparsezoo.download in your Dockerfile

#### Copying a host deployment directory

```Dockerfile
ARG MODEL_DIR
RUN mkdir /model
COPY $MODEL_DIR/deployment /model/deployment
```

#### Running sparsezoo.download in Dockerfile

```Dockerfile
RUN sparsezoo.download --save-dir /model <stub>
```

## Passing deployment directory to deepsparse

The previous steps will save the deployment to `/model/deployment` inside
your docker container.

#### deepsparse.Pipeline

When using the deepsparse.Pipeline api you would set model_path like:

```python
pipeline = Pipeline.create(model_path="/model/deployment", ...)
```

#### deepsparse server

If you are using the task subcommand:
```bash
deepsparse.server task --model_path /model/deployment
```

If you are using the config subcommand with a yaml config:
```yaml
...
endpoints:
  - ...
    model: /model/deployment
    ...
```
