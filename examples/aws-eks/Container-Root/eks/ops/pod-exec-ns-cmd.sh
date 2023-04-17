#!/bin/bash

kubectl -n $2 exec -it $(kubectl -n $2 get pods | grep $1 | cut -d ' ' -f 1) -- "${@:3}"

