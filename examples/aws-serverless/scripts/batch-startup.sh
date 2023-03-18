#!/bin/bash

region=$1
stackname=$2
imagename=$3
capabilities=CAPABILITY_NAMED_IAM

account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')
ecr_account=$account.dkr.ecr.$region.amazonaws.com
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account


# # Set the name of your Docker image for the inference script and push to ECR
local_image=inference-script
ecr_image=${account}.dkr.ecr.${region}.amazonaws.com/serverless-deepsparse:${local_image}
docker build -t ${local_image} ./batch/app_inf/
docker tag ${local_image} ${ecr_image}
docker push ${ecr_image}

cd batch
sam build
sam deploy \
    --region $region \
    --stack-name $stackname \
    --image-repository $ecr_account/$imagename \
    --capabilities $capabilities \
    --template-file ./template.yaml