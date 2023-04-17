#!/bin/bash

# Horizontal Pod Autoscalers
watch kubectl get hpa "$@"

