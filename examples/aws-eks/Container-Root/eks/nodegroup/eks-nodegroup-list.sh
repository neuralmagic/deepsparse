#!/bin/bash

echo ""

function usage() {
	echo ""
	echo "Usage: "
	echo "   $0 [CLUSTER_NAME]"
	echo ""
}

if [ "$1" != "" ]; then
	CLUSTER_NAME="$1"
fi

unset err
if [ "${CLUSTER_NAME}" == "" ]; then
	echo "Environment variable CLUSTER_NAME must be set"
	echo "or CLUSTER_NAME must be passed as argument"
	err="true"
fi

if [ "${err}" == "" ]; then
	echo ""
	echo "Listing nodegroups for cluster ${CLUSTER_NAME} ..."
	CMD="eksctl get nodegroup --cluster ${CLUSTER_NAME}"
	echo "${CMD}"
	if [ "${DRY_RUN}" == "" ]; then
		${CMD}
	fi
else
	echo "Please correct errors and try again"
	usage
fi

