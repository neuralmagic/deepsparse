#!/bin/bash

echo ""

unset err
if [ "${fargateprofile_name}" == "" ]; then
	echo "Environment variable fargateprofile_name must be set"
	err=true
fi

if [ "${CLUSTER_NAME}" == "" ]; then
	echo "Environment variable CLUSTER_NAME must be set"
	err=true
fi

if [ "${CLUSTER_REGION}" == "" ]; then
	echo "Environment variable CLUSTER_REGION must be set"
	err=true
fi

if [ "${err}" == "" ]; then
	echo ""
	echo "Creating fargate profile ${fargateprofile_name} ..."
	CMD="eksctl create fargateprofile --name ${fargateprofile_name} --namespace ${fargateprofile_name} --cluster ${CLUSTER_NAME} --region ${CLUSTER_REGION}"
	echo ${CMD}
	if [ "${DRY_RUN}" == "" ]; then
		${CMD}
	fi
else
	echo ""
	echo "Please correct any errors and try again"
fi
