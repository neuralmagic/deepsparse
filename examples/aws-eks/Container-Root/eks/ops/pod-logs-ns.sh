#!/bin/bash

kubectl -n $1 logs -f $(kubectl -n $1 get pods | grep $2 | cut -d ' ' -f 1)

