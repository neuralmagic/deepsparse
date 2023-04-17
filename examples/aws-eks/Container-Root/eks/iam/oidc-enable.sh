#!/bin/bash

if [ "$CLUSTER_NAME" == "" ]; then
	echo "Please export CLUSTER_NAME before executing this script"
else
	OIDC_PROVIDER_URL=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.identity.oidc.issuer" --output text)
	PROVIDER_ID=$(echo $OIDC_PROVIDER_URL | awk -F '/' '{print $NF}')
	PROVIDERS=$(aws iam list-open-id-connect-providers | grep ${PROVIDER_ID})
	if [ "$PROVIDERS" == "" ]; then
		echo "Associating OIDC provider $PROVIDER_ID with cluster $CLUSTER_NAME ..."
		eksctl utils associate-iam-oidc-provider --cluster $CLUSTER_NAME --approve
	else
		echo "OIDC Provider $PROVIDER_ID is already associaed with cluster $CLUSTER_NAME"
	fi
fi"
	
