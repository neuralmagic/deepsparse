#!/bin/bash

echo ""

unset err
if [ "${CLUSTER_NAME}" == "" ]; then
        echo "Environment variable CLUSTER_NAME must be set"
        err="true"
fi

if [ "${nodegroup_name}" == "" ]; then
        echo "Environment variable nodegroup_name must be set"
        err="true"
fi

if [ "${labels}" == "" ]; then
        echo "Environment variable labels must be set"
        err="true"
fi

if [ "${err}" == "" ]; then
        echo ""
        echo "Setting labels for cluster ${CLUSTER_NAME} nodegroup ${nodegroup_name} to ${labels} ..."
        CMD="eksctl set labels --cluster ${CLUSTER_NAME} --nodegroup ${nodegroup_name} --labels ${labels}"
        echo "${CMD}"
        if [ "${DRY_RUN}" == "" ]; then
                ${CMD}
        fi
else
        echo "Please correct any errors and try again"
fi

