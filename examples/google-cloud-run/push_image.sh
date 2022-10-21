#!/bin/bash

billingid=019B6C-2EF10D-D5D4E2
projectid=deepsparse-cloud-3
gcloud projects create $projectid --name="Cloud Run example"
gcloud config set project $projectid
gcloud alpha billing accounts projects link $projectid --billing-account $billingid
gcloud services enable containerregistry.googleapis.com

docker build -t sparserun .
# docker run -d --name sparse-cloud -p 80:80 sparserun
docker build . -t gcr.io/${projectid}/sparserun:latest
docker push gcr.io/${projectid}/sparserun:latest


# gcloud container images delete gcr.io/${project}/quickstart-image:tag1 --force-delete-tags

