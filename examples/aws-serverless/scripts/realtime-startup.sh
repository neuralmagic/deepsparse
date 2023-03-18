#!/bin/bash

region=$1
stackname=$2
imagename=$3
capabilities=CAPABILITY_IAM

account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')
ecr_account=$account.dkr.ecr.$region.amazonaws.com
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account

cd realtime
sam build
sam deploy \
    --region $region \
    --stack-name $stackname \
    --image-repository $ecr_account/$imagename \
    --capabilities $capabilities \
    --template-file ./template.yaml