#!/bin/bash

source ./eks.conf

echo ""
echo "Status of cluster ${CLUSTER_NAME} ..."

echo ""
CMD="eksctl get cluster --name $CLUSTER_NAME"
echo "${CMD}"
if [ "${DRY_RUN}" == "" ]; then
    ${CMD}
fi

echo ""
CMD="eksctl get nodegroups --cluster ${CLUSTER_NAME}"
echo "${CMD}"
if [ "${DRY_RUN}" == "" ]; then
    ${CMD}
fi

echo ""
CMD="eksctl get fargateprofiles --cluster ${CLUSTER_NAME}"
echo "${CMD}"
if [ "${DRY_RUN}" == "" ]; then
    ${CMD}
fi
