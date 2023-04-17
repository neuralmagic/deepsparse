#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

print_help() {
	echo ""
	echo "Usage: $0 "
	echo ""
	echo "   This script compiles/traces the model configured in config.properties and saves it locally as a .pt file"
	echo "   Tracing is supported on CPU, GPU, or Inferentia, however it must be done on a machine that has"
	echo "   the target processor chip available. Example: tracing a model for Inferentia must be done on an inf1 instance."
	echo ""
}


if [ "$1" == "" ]; then 
	source ./config.properties
	echo ""
	echo "Tracing model $huggingface_model_name ..."

	echo ""
	case "$processor" in
		"cpu")
			echo "   ... for cpu ..."
			docker run -it --rm -v $(pwd)/2-trace:/app/trace -v $(pwd)/config.properties:/app/config.properties ${registry}${base_image_name}${base_image_tag} bash -c "cd /app/trace; python model-tracer.py"
			;;
		"gpu")
			echo "   ... for gpu ..."
			docker run --gpus 0 -it --rm -v $(pwd)/2-trace:/app/trace -v $(pwd)/config.properties:/app/config.properties ${registry}${base_image_name}${base_image_tag} bash -c "cd /app/trace; python model-tracer.py"
			;;
		"inf")
			echo "   ... for inf ..."
			docker run -it --rm -e AWS_NEURON_VISIBLE_DEVICES=ALL --privileged -v $(pwd)/2-trace:/app/trace -v $(pwd)/config.properties:/app/config.properties ${registry}${base_image_name}${base_image_tag} bash -c "cd /app/trace; python model-tracer.py"
			;;
		*)
			echo "Please ensure cpu, gpu, or inf is configure as processor in config.properties"
			exit 1
			;;
	esac
else
	print_help
fi

