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
    server=0
    while [ $server -lt $num_servers ]; do
        CMD="docker rm -f ${app_name}-${server}"
        echo "$CMD"
        eval "$CMD"
	server=$((server+1))
    done
elif [ "$runtime" == "kubernetes" ]; then
    kubectl delete -f ${app_dir} 
else
    echo "Runtime $runtime not recognized"
fi
