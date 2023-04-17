#!/bin/bash

source ./eks.conf

echo ""
date
echo "Scaling cluster ${CLUSTER_NAME} ..."

# Scale CPU nodegroups
for index in ${!CPU_NODEGROUP_INSTANCE_TYPES[@]}
do
	export instance_type=${CPU_NODEGROUP_INSTANCE_TYPES[$index]}
	export nodegroup_name=$(echo ${instance_type} | sed -e 's/\./-/g')
	export nodegroup_size=${CPU_NODEGROUP_SIZES[$index]}
	nodegroup/eks-nodegroup-scale.sh
done 

# Scale GPU nodegroups
for index in ${!GPU_NODEGROUP_INSTANCE_TYPES[@]}
do
        export instance_type=${GPU_NODEGROUP_INSTANCE_TYPES[$index]}
        export nodegroup_name=$(echo ${instance_type} | sed -e 's/\./-/g')
        export nodegroup_size=${GPU_NODEGROUP_SIZES[$index]}
        nodegroup/eks-nodegroup-scale.sh
done

# Scale ASIC nodegroups
for index in ${!ASIC_NODEGROUP_INSTANCE_TYPES[@]}
do
        export instance_type=${ASIC_NODEGROUP_INSTANCE_TYPES[$index]}
        export nodegroup_name=$(echo ${instance_type} | sed -e 's/\./-/g')
        export nodegroup_size=${ASIC_NODEGROUP_SIZES[$index]}
        nodegroup/eks-nodegroup-scale.sh
done

echo ""
date
echo "Done scaling cluster ${CLUSTER_NAME}"
echo ""

