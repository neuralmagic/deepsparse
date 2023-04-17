#!/bin/bash

source eks.conf

USER_ARN=$(aws sts get-caller-identity --output json | jq -r '.Arn')
USER_NAME=$(echo $USER_ARN | cut -d '/' -f 2)

eksctl create iamidentitymapping --cluster $CLUSTER_NAME --arn $USER_ARN --group system:masters --username $USER_NAME

