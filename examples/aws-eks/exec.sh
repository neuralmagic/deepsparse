#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

source .env

if [ "$1" == "" ]; then
	CMD="bash"
else
	CMD=$@
fi

docker container exec -it ${CONTAINER} $CMD 

