#!/bin/bash

aws cloudformation create-stack --stack-name ManagementInstance --template-body file://ManagementInstance.json --capabilities CAPABILITY_IAM

