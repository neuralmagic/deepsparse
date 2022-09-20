#!/bin/bash

account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')
region=$(aws configure get region)
stack_name=lambda-stack
image_name=/lambda-deepsparse
capabilities=CAPABILITY_IAM
ecr_account=${account}.dkr.ecr.${region}.amazonaws.com
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account

cd lambda-deepsparse
sam build
sam deploy \
    --region $region \
    --stack-name $stack_name \
    --image-repository $ecr_account$image_name \
    --capabilities $capabilities
