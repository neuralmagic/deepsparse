#!/bin/bash

. ./eks.conf

aws eks update-kubeconfig --name $CLUSTER_NAME
