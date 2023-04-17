#!/bin/bash

#curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
#chmod 700 get_helm.sh
#./get_helm.sh
#rm -f ./get_helm.sh

curl -L https://git.io/get_helm.sh | bash -s -- --version v3.11.2

helm version

