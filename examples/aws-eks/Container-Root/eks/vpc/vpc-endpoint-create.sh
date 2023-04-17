#!/bin/bash
#This script requires an up-to-date version of the aws cli tool


get_vpc_route_table_ids () {
   # local profile=$1
    local vpc=$1
    local routetable_id
    routetable_id=$(aws ec2 \
        describe-route-tables \
        --filters "Name=vpc-id,Values=$vpc" \
        --query '*[][].RouteTableId' --output text | tr '[:blank:]' ',')
    echo $routetable_id
}

create_vpc_end_point () {
    local vpc_id=$1
    local service_name=$2
    local route_table_ids=$3

    route_table_ids=$(echo $3 | tr ',' ' ')

    aws ec2 create-vpc-endpoint \
        --vpc-id $vpc_id \
        --service-name $service_name \
        --route-table-ids $route_table_ids
}

if [ "$CLUSTER_NAME" == "" ]; then
	echo "Please export CLUSTER_NAME before running this script"
else
	if [ "$REGION" == "" ]; then
		echo "Please export REGION before running this script"
	else
		echo "Creating VPC endpoint ..."
		service_name=com.amazonaws.${REGION}.s3
		export VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
		route_table_ids=$(get_vpc_route_table_ids $VPC_ID)
		create_vpc_end_point $VPC_ID $service_name $route_table_ids
	fi
fi

