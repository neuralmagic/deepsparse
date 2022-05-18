#!/bin/bash

account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')
region=$(aws configure get region)
ecr_account=${account}.dkr.ecr.${region}.amazonaws.com

aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account
fullname=$ecr_account/deepsparse-sagemaker:latest

docker tag deepsparse-sagemaker-example:latest $fullname
docker push $fullname