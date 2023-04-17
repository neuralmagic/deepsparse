#!/bin/bash

cmd=bash
if [ ! "$2" == "" ]; then
	cmd="$2"
fi

kubectl exec -it $(kubectl get pods | grep $1 | cut -d ' ' -f 1) -- $cmd

