#!/bin/bash

region=$1
stackname=$2
imagename=$3

# Set the IAM capabilities for the deployment
capabilities=CAPABILITY_NAMED_IAM

# Set the name of your Docker image for the inference script
inference_image=inference-script

# Get the AWS account ID
account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')

# Set the ECR repository name
ecr_account=$account.dkr.ecr.$region.amazonaws.com

# Log in to the ECR repository
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account

# Set the name of your Docker image for the inference script and push to ECR
ecr_image=${account}.dkr.ecr.${region}.amazonaws.com/serverless-deepsparse:${inference_image}

# Build the Docker image for the inference script
docker build -t ${inference_image} ./batch/app_inf/

# Tag the Docker image and push to ECR
docker tag ${inference_image} ${ecr_image}
docker push ${ecr_image}

# Change directory to the 'batch' folder
cd batch

# Build the SAM application
sam build

# Deploy the SAM application
sam deploy \
    --region $region \
    --stack-name $stackname \
    --image-repository $ecr_account/$imagename \
    --capabilities $capabilities \
    --template-file ./template.yaml
