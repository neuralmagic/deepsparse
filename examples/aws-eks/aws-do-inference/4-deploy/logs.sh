#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

if [ -f ../config.properties ]; then
    source ../config.properties
elif [ -f ./config.properties ]; then
    source ./config.properties
else
    echo "config.properties not found!"
fi

echo ""
echo "Runtime: $runtime"
echo "Processor: $processor"

if [ "$runtime" == "docker" ]; then
    if [ "$num_servers" == "1" ]; then
        CMD="docker logs -f ${app_name}-0"
    else
        if [ "$1" == "" ]; then
            CMD="docker ps | grep ${app_name}- | cut -d ' ' -f 1 | xargs -L 1 docker logs"
        else
            CMD="docker logs -f ${app_name}-$1"
        fi
    fi
    echo "$CMD"
    eval "$CMD"
elif [ "$runtime" == "kubernetes" ]; then
    command -v kubetail > /dev/null
    if [ "$?" == "1" ]; then
        echo "kubetail not found"
        echo "Please follow the instructions here https://github.com/johanhaleby/kubetail#installation, then try again"
    else
        if [ "$1" == "" ]; then
            kubetail -n ${namespace} -f ${app_name}
        else
            kubectl -n ${namespace} logs -f $(kubectl -n ${namespace} get pods | grep ${app_name}-$1 | cut -d ' ' -f 1)
        fi
    fi
else
    echo "Runtime $runtime not recognized"
fi