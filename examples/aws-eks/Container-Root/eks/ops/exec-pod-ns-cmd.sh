#!/bin/bash

kubectl -n $1 exec -it $(kubectl -n $1 get pods | grep $2 | cut -d ' ' -f 1) -- "${@:3}"

