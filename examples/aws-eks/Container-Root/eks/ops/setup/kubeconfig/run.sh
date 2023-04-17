#!/bin/bash

kubectl apply -f ./sa.yaml

./get-sa-kubeconfig.sh sa-admin kube-system

