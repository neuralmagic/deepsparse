#!/bin/bash

kubectl -n $1 delete pod $(kubectl -n $1 get pods | cut -d ' ' -f 1) 

