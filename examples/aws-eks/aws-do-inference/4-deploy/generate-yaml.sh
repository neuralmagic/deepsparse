#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

set -a

if [ -f ../config.properties ]; then
    source ../config.properties
elif [ -f ./config.properties ]; then
    source ./config.properties
else
    echo "config.properties not found!"
fi

template=./${processor}-yaml.template
prefix=${app_name}-
instance_start=0
instances=${num_servers}

if [ -d ./${app_dir} ]; then
    rm -rf ./${app_dir}
fi
mkdir -p ./${app_dir}

instance=$instance_start
while [ $instance -lt $instances ]
do
	export instance_name=${prefix}${instance}
	echo "Generating ./${app_dir}/${instance_name}.yaml ..."
	CMD="cat $template | envsubst > ./${app_dir}/${instance_name}.yaml"
	#echo "$CMD"
	eval "$CMD"
	instance=$((instance+1))
done

set +a