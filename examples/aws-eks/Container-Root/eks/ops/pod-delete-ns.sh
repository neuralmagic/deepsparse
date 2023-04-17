#!/bin/bash

kubectl -n $2 delete pod $(kubectl -n $2 get pods | grep $1 | cut -d ' ' -f 1) 

