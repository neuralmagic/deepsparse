#!/bin/sh

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

# Container startup script
echo "Container-Root/startup.sh executed"

cd /eks

while true; do echo do-eks container is running at $(date); sleep 10; done

