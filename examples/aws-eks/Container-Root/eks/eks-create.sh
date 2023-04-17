#!/bin/bash

source ./eks.conf

if [ "$CONFIG" == "conf" ]; then

	# Create EKS Cluster with initial nodegroup to run kube-system pods
	echo ""
	date
	echo "Creating cluster ${CLUSTER_NAME} ..."
	CMD="eksctl create cluster --name ${CLUSTER_NAME} --region ${CLUSTER_REGION} --version ${CLUSTER_K8S_VERSION} \
	--zones "${CLUSTER_ZONES}" --vpc-cidr ${CLUSTER_VPC_CIDR} ${CLUSTER_OPTIONS}"
	echo "${CMD}"
	if [ "${DRY_RUN}" == "" ]; then
		${CMD}
	fi

	# Create CPU nodegroups
	echo ""
	echo "Creating CPU nodegroups in cluster ${CLUSTER_NAME} ..."
	export nodegroup_opts="${CPU_NODEGROUP_OPTIONS}"
	for index in ${!CPU_NODEGROUP_INSTANCE_TYPES[@]}
	do
		export instance_type=${CPU_NODEGROUP_INSTANCE_TYPES[$index]}
		export nodegroup_name=$(echo $instance_type | sed -e 's/\./-/g')
		nodegroup/eks-nodegroup-create.sh
	done

	# # Create GPU nodegroups
	# echo ""
	# echo "Creating GPU nodegroups in cluster ${CLUSTER_NAME} ..."
	# export nodegroup_opts="${GPU_NODEGROUP_OPTIONS}"
	# for index in ${!GPU_NODEGROUP_INSTANCE_TYPES[@]}
	# do
	# 	export instance_type=${GPU_NODEGROUP_INSTANCE_TYPES[$index]}
	# 	export nodegroup_name=$(echo $instance_type | sed -e 's/\./-/g')
	# 	nodegroup/eks-nodegroup-create.sh
	# done

	# # Create ASIC nodegroups
	# echo ""
	# echo "Creating ASIC nodegroups in cluster ${CLUSTER_NAME} ..."
	# export nodegroup_opts="${ASIC_NODEGROUP_OPTIONS}"
	# for index in ${!ASIC_NODEGROUP_INSTANCE_TYPES[@]}
	# do
	# 	export instance_type=${ASIC_NODEGROUP_INSTANCE_TYPES[$index]}
	# 	export nodegroup_name=$(echo $instance_type | sed -e 's/\./-/g')
	# 	nodegroup/eks-nodegroup-create.sh
	# done

	# Create Fargate Profiles
	echo ""
	echo "Creating Fargate Profiles in cluster ${CLUSTER_NAME} ..."
	for index in ${!SERVERLESS_FARGATE_PROFILE_NAMES}
	do
		export fargateprofile_name=${SERVERLESS_FARGATE_PROFILE_NAMES[$index]}
		fargateprofile/eks-fargateprofile-create.sh
	done

	# Scale cluster as specified
	./eks-scale.sh

	# Optionally deploy cluster autoscaler
	if [ "$CLUSTER_AUTOSCALER_DEPLOY" == "true" ]; then
		pushd deployment/cluster-autoscaler
		./deploy-cluster-autoscaler.sh
		popd
	fi

	# Done creating EKS Cluster
	echo ""
	date
	echo "Done creating cluster ${CLUSTER_NAME}"
	echo ""

elif [ "$CONFIG" == "yaml" ]; then
	echo ""
	echo "Creating cluster using eks.yaml ..."
	CMD="eksctl create cluster -f ${EKS_YAML}"
        echo "${CMD}"
        if [ "${DRY_RUN}" == "" ]; then
                ${CMD}
        fi
	echo ""
	echo "Done creating cluster using ${EKS_YAML}"
	echo ""
else
	echo ""
	echo "Unrecognized CONFIG type $CONFIG. Please specify CONFIG=conf or CONFIG=yaml in eks.conf"
	echo ""
fi
