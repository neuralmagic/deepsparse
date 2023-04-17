#!/bin/bash 

source ../eks.conf

# Desired fargateprofiles
desired_fargateprofiles=( ${SERVERLESS_FARGATE_PROFILE_NAMES[@]} )
echo ""
echo "Desired Fargate profiles:"
echo ${desired_fargateprofiles[@]}

# Current fargateprofiles
fpout=$(eksctl get fargateprofiles --cluster $CLUSTER_NAME --output json)
fp_names=$(echo $fpout | jq '.[].name')

current_fargateprofiles=()
for fpname in $fp_names
do
	fp=$(echo $fpname | sed -e 's/\"//g')
	current_fargateprofiles=( ${current_fargateprofiles[@]} $fp )
done
echo ""
echo "Current Fargate profiles:"
echo ${current_fargateprofiles[@]}

# Differences
for desired_index in ${!desired_fargateprofiles[@]}
do
        desired_fargateprofile=${desired_fargateprofiles[$desired_index]}
        for current_index in ${!current_fargateprofiles[@]}
        do
                current_fargateprofile=${current_fargateprofiles[$current_index]}
                if [ "$desired_fargateprofile" == "$current_fargateprofile" ]; then
                        desired_fargateprofiles[$desired_index]=""
                        current_fargateprofiles[$current_index]=""
                        break
                fi
        done
done

echo ""

# Add new Fargate profiles
echo "Fargate profiles to add:"
new_profiles=$(echo ${desired_fargateprofiles[@]})
echo "${new_profiles[@]}"
for index in ${!new_profiles[@]}
do
	export fargateprofile_name=${new_profiles[$index]}
	if [ ! "$fargateprofile_name" == "" ]; then
		./do-eks-fargateprofile-create.sh
	fi
done

# Delete old Fargate profiles
echo "Fargate profiles to remove:"
old_profiles=$(echo ${current_fargateprofiles[@]})
echo "${old_profiles[@]}"
for index in ${!old_profiles[@]}
do
	export fargateprofile_name=${old_profiles[$index]}
	if [ ! "$fargateprofile_name" == "" ]; then
		./do-eks-fargateprofile-delete.sh
	fi
done

