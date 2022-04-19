## DeepSparse docker image
This directory contains the Dockerfile to create minimal DeepSparse docker image.
This image is based off the latest official Ubuntu image.

In order to build this image, run from the root directory:
`docker build -t deepsparse_docker . && docker run -it deepsparse_docker`
