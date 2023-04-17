#!/bin/sh

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

if [ -d /etc/apt ]; then
        [ -n "$http_proxy" ] && echo "Acquire::http::proxy \"${http_proxy}\";" > /etc/apt/apt.conf; \
        [ -n "$https_proxy" ] && echo "Acquire::https::proxy \"${https_proxy}\";" >> /etc/apt/apt.conf; \
        [ -f /etc/apt/apt.conf ] && cat /etc/apt/apt.conf
fi

# Install basic tools
apt-get update && apt-get install -y curl jq vim nano less unzip git gettext-base groff sudo htop bash-completion

# Install aws cli
./eks/ops/setup/install-aws-cli.sh

# Install eksctl
./eks/ops/setup/install-eksctl.sh

# Install kubectl
./eks/ops/setup/install-kubectl.sh

# Install kubectx
./eks/ops/setup/install-kubectx.sh

# Install kubetail
./eks/ops/setup/install-kubetail.sh

# Install kubeshell
./eks/ops/setup/install-kubeshell.sh

# Install helm
./eks/ops/setup/install-helm.sh

# Install docker
./eks/ops/setup/install-docker.sh

# Install golang
./eks/ops/setup/install-go.sh

# Install monitui
./eks/ops/setup/install-monitui.sh

# Install python
./eks/ops/setup/install-python.sh
python -m pip install torchx[kubernetes]

# Install kubeps1 and configure bashrc aliases 
./eks/ops/setup/install-kubeps1.sh

