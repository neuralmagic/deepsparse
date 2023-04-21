#!/bin/bash

region=$1
stackname=$2
imagename=$3

# Set the IAM capabilities for the deployment
capabilities=CAPABILITY_NAMED_IAM

# Get the AWS account ID
account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')

# Set the ECR repository name
ecr_account=$account.dkr.ecr.$region.amazonaws.com

# Log in to the ECR repository
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account

# Change directory to the 'realtime' folder
cd realtime

# Build the SAM application
sam build

# Deploy the SAM application
sam deploy \
    --region $region \
    --stack-name $stackname \
    --image-repository $ecr_account/$imagename \
    --capabilities $capabilities \
    --template-file ./template.yaml
