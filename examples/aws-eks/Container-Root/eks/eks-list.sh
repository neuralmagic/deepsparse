#!/bin/bash

source ./eks.conf

echo ""
echo "List of EKS clusters ..."

echo ""
CMD="eksctl get cluster"
echo "${CMD}"
if [ "${DRY_RUN}" == "" ]; then
    ${CMD}
fi

