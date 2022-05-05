## DeepSparse docker image
This directory contains the Dockerfile to create minimal DeepSparse docker image.
This image is based off the latest official Ubuntu image.

In order to build and launch this image, run from the root directory:
`docker build -t deepsparse_docker . && docker run -it deepsparse_docker ${python_command}`, for example:

`docker build -t deepsparse_docker . && docker run -it deepsparse_docker deepsparse.server --task question_answering --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"`
