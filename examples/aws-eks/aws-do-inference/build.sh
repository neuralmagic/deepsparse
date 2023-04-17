#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

print_help() {
	echo ""
	echo "Usage: $0 [arg]"
	echo ""
	echo "   When no arguments are specified, this script builds a base container "
	echo "   for the processor type, configured in config.properties."
	echo "   Optionally, the script can push/pull the base image to/from a container registry."
	echo ""
	echo "   Available optional arguments:"
	echo "      push   - push base image to container registry"
	echo "      pull   - pull base image from container registry"
        echo ""	
}


action=$1
if [ "$action" == "" ]; then
	source ./config.properties

	echo ""
	echo "Building base container ..."
	
	echo ""
	case "$processor" in
		"cpu")
			echo "   ... base-cpu ..."
			docker build -t ${registry}${base_image_name}${base_image_tag} -f ./1-build/Dockerfile-base-cpu .
			;;
		"gpu")
			echo "   ... base-gpu ..."
			docker build -t ${registry}${base_image_name}${base_image_tag} -f ./1-build/Dockerfile-base-gpu .
			;;
		"inf")
			echo "   ... base-inf ..."
			docker build -t ${registry}${base_image_name}${base_image_tag} -f ./1-build/Dockerfile-base-inf .
			;;
		*)
			echo "Please ensure cpu, gpu, or inf is configure as processor in config.properties"
			exit 1
			;;
	esac

elif [ "$action" == "push" ]; then
	./1-build/push.sh
elif [ "$action" == "pull" ]; then
	./-build/pull.sh
else 
	print_help
fi
