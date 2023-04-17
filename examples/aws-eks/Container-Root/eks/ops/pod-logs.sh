#!/bin/bash

kubectl logs -f $(kubectl get pods | grep $1 | cut -d ' ' -f 1)

