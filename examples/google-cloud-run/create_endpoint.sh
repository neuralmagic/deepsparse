#!/bin/bash

billingid=$1
projectid=$2
image_name=$3
region_name=$4

# creates GCP project
gcloud projects create $projectid --name="Cloud Run example"
# set GCP project
gcloud config set project $projectid
# link GCP billing account with projectid
gcloud alpha billing accounts projects link $projectid --billing-account $billingid
# enable container registry API
gcloud services enable containerregistry.googleapis.com
# enable cloud run API
gcloud services enable run.googleapis.com

# build image and push to container registry on GCP
docker build -t $image_name .
docker build . -t gcr.io/$projectid/$image_name:latest
docker push gcr.io/$projectid/$image_name:latest

# deploy image on Cloud Run with deployment configuration
gcloud run deploy deepsparse-cloudrun \
    --image gcr.io/$projectid/$image_name:latest \
    --region $region_name \
    --min-instances 0 \
    --max-instances 2 \
    --cpu 2 \
    --concurrency 1 \
    --memory 3Gi \
    --timeout 2m \
    --port 80 \
    --allow-unauthenticated
    
# TESTING #
# docker run -d --name sparse-cloud -p 80:80 sparserun
# gcloud container images delete gcr.io/${project}/quickstart-image:tag1 --force-delete-tags
