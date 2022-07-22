# DeepSparse docker image

This directory contains the Dockerfile to create minimal DeepSparse docker image.
This image is based off the latest official Ubuntu image.

## Pull
You can access the already built image detailed at https://github.com/orgs/neuralmagic/packages/container/package/deepsparse:

```bash
docker pull ghcr.io/neuralmagic/deepsparse:1.0.2-debian11
```

## Extend
If you would like to customize the docker image, you can use the pre-built images as a base in your own `Dockerfile`:

```Dockerfile
from ghcr.io/neuralmagic/deepsparse:1.0.2-debian11

...
```

## Build
In order to build and launch this image, run from the root directory:
`docker build -t deepsparse_docker . && docker run -it deepsparse_docker ${python_command}`, for example:

`docker build -t deepsparse_docker . && docker run -it deepsparse_docker deepsparse.server --task question_answering --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"`

If you want to use a specific branch from deepsparse you can use the `GIT_CHECKOUT` build arg:
```
docker build --build-arg GIT_CHECKOUT=main -t deepsparse_docker .
```
