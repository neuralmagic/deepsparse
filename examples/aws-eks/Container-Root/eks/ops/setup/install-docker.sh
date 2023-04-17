#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

if [ -f /usr/bin/apt-get ]; then
    apt-get remove docker docker-engine docker.io containerd runc
    apt-get update
    apt-get install -y ca-certificates curl gnupg lsb-release apt-transport-https software-properties-common iproute2
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce-cli
    #apt-get install -y docker-ce docker-ce-cli containerd.io
else
    echo "/usr/bin/apt-get does not exist"
    echo "Cannot install Docker cli with this script"
    echo "Please refer to https://docs.docker.com/engine/install"
fi

