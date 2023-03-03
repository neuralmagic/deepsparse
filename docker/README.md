# DeepSparse docker image

This directory contains the Dockerfile to create DeepSparse docker image.
This image is based off the latest official Python:3.8.16 image

## Pull
You can access the already built image detailed at https://github.com/orgs/neuralmagic/packages/container/package/deepsparse:

```bash
docker pull ghcr.io/neuralmagic/deepsparse:1.4
docker tag ghcr.io/neuralmagic/deepsparse:1.4 deepsparse_docker
```

## Extend
If you would like to customize the docker image, you can use the pre-built images as a base in your own `Dockerfile`:

```Dockerfile
FROM ghcr.io/neuralmagic/deepsparse:1.4
...
```

## Build
In order to build and launch this image, run from the root directory:
`docker build -t deepsparse_docker . && docker run -it deepsparse_docker ${python_command}`, for example:

`docker build -t deepsparse_docker . && docker run -it deepsparse_docker deepsparse.server --task question_answering --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"`

If you want to use a specific branch from deepsparse you can use the `GIT_CHECKOUT` build arg:
```
docker build --build-arg BRANCH=main -t deepsparse_docker .
```


We also provide the following pre-built images with all integration specific 
dependencies installed:

| Image Name                	| Description                                                            	|
|---------------------------	|------------------------------------------------------------------------	|
| deepsparse-base           	| Base DeepSparse container with no integration specific dependencies    	|
| deepsparse/deepsparse-all 	| DeepSparse container with all major integration dependencies installed 	|
| deepsparse-server         	| DeepSparse container with `[server]` dependencies installed              	|
| deepsparse-transformers   	| DeepSparse container with all transformer dependencies installed       	|
| deepsparse-torchvision    	| DeepSparse container with torchvision dependencies installed           	|
| deepsparse-ultralytics    	|  DeepSparse container with yolov5 and yolov8 dependencies installed    	|


To build a development image for a specific branch use the  following  command:

```bash
docker build \
  --build-arg BRANCH=[BRANCH_NAME] \
  --build-arg DEPS=dev \
  -t deepsparse_docker .
```

To run the container:

```bash
docker container run -it deepsparse_docker
```