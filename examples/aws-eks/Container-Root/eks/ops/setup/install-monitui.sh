#!/bin/bash

# Install eks-node-viewer (f.k.a monitui)
# https://github.com/awslabs/eks-node-viewer/

source ~/.bashrc
go env -w GOPROXY=direct
go install github.com/awslabs/eks-node-viewer/cmd/eks-node-viewer@latest
export GOBIN=${GOBIN:-~/go/bin}
echo "export PATH=\$PATH:$GOBIN" >> ~/.bashrc

