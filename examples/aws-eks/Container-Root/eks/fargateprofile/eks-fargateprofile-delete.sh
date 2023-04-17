#!/bin/bash

echo ""

unset err
if [ "${CLUSTER_NAME}" == "" ]; then
	echo "Environment variable CLUSTER_NAME must be set"
	err="true"
fi

if [ "${fargateprofile_name}" == "" ]; then
	echo "Environment variable fargateprofile_name must be set"
	err="true"
fi

#if [ "${CLUSTER_REGION}" == "" ]; then
#	echo "Environment variable CLUSTER_REGION must be set"
#	err="true"
#fi

if [ "${err}" == "" ]; then
	echo ""
	echo "Deleting Fargate profile ${fargateprofile_name} ..."
	CMD="eksctl delete fargateprofile --cluster ${CLUSTER_NAME} --name ${fargateprofile_name}"
	echo "${CMD}"
	if [ "${DRY_RUN}" == "" ]; then
		${CMD}
	fi
else
	echo "Please correct any errors and try again"
fi
