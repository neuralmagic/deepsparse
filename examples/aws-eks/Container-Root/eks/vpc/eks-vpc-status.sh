#!/bin/bash

source ../eks.conf

echo ""
echo "Describing VPCs ... "

unset err
if [ "${CLUSTER_REGION}" == "" ]; then
    echo "Environment variable CLUSTER_REGION must be set"
    err="true"
fi

if [ "${err}" == "" ]; then
    CMD="aws ec2 describe-vpcs --region ${CLUSTER_REGION} --output text"
    echo "${CMD}"
    if [ "${DRY_RUN}" == "" ]; then
        ${CMD}
    fi
else
    echo "Please resolve any errors and try again"
fi
