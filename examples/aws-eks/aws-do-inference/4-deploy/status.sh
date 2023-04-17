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
    CMD="docker ps -a | grep ${app_name}"
    echo "$CMD"
    eval "$CMD"
elif [ "$runtime" == "kubernetes" ]; then
    if [ "$1" == "" ]; then
        echo ""
        echo "Pods:"
        kubectl -n ${namespace} get pods
        echo ""
        echo "Services:"
        kubectl -n ${namespace} get services
    else
        echo ""
        echo "Pod:"
        kubectl -n ${namespace} get pod $(kubectl -n ${namespace} get pods | grep ${app_name}-$1 | cut -d ' ' -f 1) -o wide
        echo ""
        echo "Service:"
        kubectl -n ${namespace} get service ${app_name}-$1
    fi
else
    echo "Runtime $runtime not recognized"
fi