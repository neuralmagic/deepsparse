#!/bin/bash

kubectl delete pod $(kubectl get pods | grep $1 | cut -d ' ' -f 1) 

