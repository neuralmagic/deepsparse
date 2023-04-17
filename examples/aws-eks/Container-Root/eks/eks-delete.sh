#!/bin/bash

source ./eks.conf

if [ "$CONFIG" == "conf" ]; then

	echo ""
	echo "Deleting cluster ${CLUSTER_NAME} ..."

	CMD="eksctl delete cluster --name ${CLUSTER_NAME}"
elif [ "$CONFIG" == "yaml" ]; then
	echo ""
	echo "Deleting cluster using ${EKS_YAML} ..."
	
	CMD="eksctl delete cluster -f ${EKS_YAML}"
else
	echo ""
	echo "Unrecognized CONFIG type $CONFIG"
	echo "Please specify CONFIG=conf or CONFIG=yaml in eks.conf"
	echo ""
	exit 1
fi

echo ${CMD}
if [ "${DRY_RUN}" == "" ]; then
    ${CMD}
fi
