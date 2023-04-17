#!/bin/bash

kubectl get nodes -o yaml "$@" | grep instance-type | grep node | grep -v f:

