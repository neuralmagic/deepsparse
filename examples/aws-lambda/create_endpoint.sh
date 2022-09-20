#!/bin/bash

account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')
capabilities=CAPABILITY_IAM
ecr_account=${account}.dkr.ecr.$1.amazonaws.com
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account

cd lambda-deepsparse
sam build
sam deploy \
    --region $1 \
    --stack-name $2 \
    --image-repository $ecr_account/$3 \
    --capabilities $capabilities