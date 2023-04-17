#!/bin/bash

if [ "${CLUSTER_NAME}" = "" ]; then
        echo "Please export CLUSTER_NAME before running this script"
else
        echo "Attaching policies to node role for cluster ${CLUSTER_NAME} ..."
        export NODE_IAM_ROLE_NAME=$(eksctl get iamidentitymapping --cluster ${CLUSTER_NAME} | grep  arn | awk  '{print $1}' | egrep -o eks.*)
        aws iam attach-role-policy --role-name ${NODE_IAM_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        aws iam attach-role-policy --role-name ${NODE_IAM_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
        aws iam attach-role-policy --role-name ${NODE_IAM_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
        aws iam attach-role-policy --role-name ${NODE_IAM_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/AmazonElasticFileSystemFullAccess
	aws iam attach-role-policy --role-name ${NODE_IAM_ROLE_NAME} --policy-arn arn:aws:iam::aws:policy/IAMReadOnlyAccess
fi

