#!/bin/bash

source ../eks.conf

# Desired nodegroups
desired_nodegroup_names=()
desired_instance_types=( ${CPU_NODEGROUP_INSTANCE_TYPES[@]} ${GPU_NODEGROUP_INSTANCE_TYPES[@]} ${ASIC_NODEGROUP_INSTANCE_TYPES[@]} ${CLUSTER_SYSTEM_NODEGROUP_NAME} )
echo "Desired instance types:"
echo ${desired_instance_types[@]}
for index in ${!desired_instance_types[@]}
do
	desired_instance_type=${desired_instance_types[$index]}
	desired_nodegroup_name=$(echo $desired_instance_type | sed -e 's/\./-/g')
	desired_nodegroup_names=( ${desired_nodegroup_names[@]} $desired_nodegroup_name )
done
echo "Desired nodegroups:"
echo ${desired_nodegroup_names[@]}

# Current nodegroups
ngsout=$(eksctl get nodegroups --cluster $CLUSTER_NAME --output json)
ngsclean=$(echo $ngsout | cut -d '{' --complement -s -f 1)
ngsjson="[ { ${ngsclean}"
ngs=$(echo $ngsjson | jq '.[].Name')
current_nodegroup_names=()
for ng in $ngs
do
	nodegroup_name=$(echo $ng | sed -e 's/\"//g')
	current_nodegroup_names=( ${current_nodegroup_names[@]} $nodegroup_name )
done
echo "Current nodegroups:"
echo ${current_nodegroup_names[@]}

# Differences
for desired_index in ${!desired_nodegroup_names[@]}
do
	desired_nodegroup_name=${desired_nodegroup_names[$desired_index]}
	for current_index in ${!current_nodegroup_names[@]}
	do
		current_nodegroup_name=${current_nodegroup_names[$current_index]}
		if [ "$desired_nodegroup_name" == "$current_nodegroup_name" ]; then
			desired_nodegroup_names[$desired_index]=""
			current_nodegroup_names[$current_index]=""
			break
		fi
	done
done

echo ""

# Add new node groups 
echo "Groups to add:"
new_groups=$(echo ${desired_nodegroup_names[@]})
echo ${new_groups[@]}
for index in ${!new_groups[@]}
do
	new_group=${new_groups[$index]}
	export nodegroup_opts="${CPU_NODEGROUP_OPTIONS}"
	# Check if group name is in GPU and adjust nodegroup_opts accordingly
	new_group_is_gpu="false"
	for gpu_index in ${!GPU_NODEGROUP_INSTANCE_TYPES[@]}
	do
		gpu_instance_type=${GPU_NODEGROUP_INSTANCE_TYPES[$gpu_index]}
		gpu_nodegroup_name=$(echo $gpu_instance_type | sed -e 's/\./-/g')
		if [ "$new_group" == "$gpu_nodegroup_name" ]; then
			new_group_is_gpu="true"
			export nodegroup_opts="${GPU_NODEGROUP_OPTIONS}"
			break
		fi
	done
	# Check if group name is in ASIC array and adjust nodegroup_opts accordingly
	if [ "$new_group_is_gpu" == "false" ]; then
        	for asic_index in ${!ASIC_NODEGROUP_INSTANCE_TYPES[@]}
        	do      
                	asic_instance_type=${ASIC_NODEGROUP_INSTANCE_TYPES[$asic_index]}
                	asic_nodegroup_name=$(echo $asic_instance_type | sed -e 's/\./-/g')
                	if [ "$new_group" == "$asic_nodegroup_name" ]; then
                        	export nodegroup_opts="${ASIC_NODEGROUP_OPTIONS}"
                        	break
                	fi
        	done
	fi
	export nodegroup_name=${new_group}
	if [ "$nodegroup_name" == "${CLUSTER_SYSTEM_NODEGROUP_NAME}" ]; then
		export instance_type=${CLUSTER_SYSTEM_NODEGROUP_INSTANCE_TYPE}
		export nodegroup_opts=${CLUSTER_SYSTEM_NODEGROUP_OPTIONS}
	else
		export instance_type=$(echo $new_group | sed -e 's/-/\./g')
	fi
	if [ ! "$nodegroup_name" == "" ]; then
		./do-eks-nodegroup-create.sh
	fi
done

# Remove deleted node groups
echo "Groups to remove:"
old_groups=$(echo ${current_nodegroup_names[@]})
echo ${old_groups[@]}
for index in ${!old_groups[@]}
do
	old_group=${old_groups[$index]}
	export nodegroup_name=${old_group}
	if [ ! "$nodegroup_name" == "" ]; then
		./do-eks-nodegroup-delete.sh	
	fi
done

