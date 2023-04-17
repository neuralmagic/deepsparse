#!/bin/bash

kubectl describe pod $(kubectl get pods | grep $1 | cut -d ' ' -f 1) $2 $3 $4 $5

