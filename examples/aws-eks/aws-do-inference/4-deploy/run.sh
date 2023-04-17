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
	    run_opts="--name ${app_name}-${server} -e NUM_MODELS=$num_models -e POSTPROCESS=$postprocess -e QUIET=$quiet -P -v $(pwd)/../3-pack:/app/dev"    
    	if [ "$processor" == "gpu" ]; then
            run_opts="--gpus 0 ${run_opts}"
    	fi
	if [ "$processor" == "inf" ]; then
	    run_opts="--device=/dev/neuron${server} ${run_opts}"
	fi
    	CMD="docker run -d ${run_opts} ${registry}${model_image_name}${model_image_tag}"
    	echo "$CMD"
    	eval "$CMD"
	server=$((server+1))
    done
elif [ "$runtime" == "kubernetes" ]; then
    kubectl create namespace ${namespace} --dry-run=client -o yaml | kubectl apply -f -
    ./generate-yaml.sh
    kubectl apply -f ${app_dir}
else
    echo "Runtime $runtime not recognized"
fi
