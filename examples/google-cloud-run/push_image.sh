#!/bin/bash

docker build -t sparserun .
docker run -n sparse-cloud -p 80:80 sparserun
docker build . -t gcr.io/sparse-cloud/sparserun:latest
docker push gcr.io/sparse-cloud/sparserun:latest
