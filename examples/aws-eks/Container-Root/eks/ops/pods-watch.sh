#!/bin/bash

watch kubectl get pods -o wide "$@"

