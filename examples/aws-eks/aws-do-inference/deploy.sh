#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

print_help() {
    echo ""
    echo "Usage: $0 [arg]"
    echo ""
    echo "   This script deploys and manages the model servers on the configured runtime."
    echo "   Both Docker for single host deployments and Kubernetes for cluster deployments are supported container runtimes."
    echo "   If no arguments are specified, the default action (run) will be executed."
    echo ""
    echo "   Available optional arguments:"
    echo "   run         - deploy the model servers to the configured runtime."
    echo "   stop        - remove the model servers from the configured runtime."
    echo "   status [id] - show current status of deployed model servers, optionally just for the specified server id."
    echo "   logs [id]   - show model server logs for all servers, or only the specified server id."
    echo "   exec <id>   - open bash shell into the container of the server with the specified id. Note the id is required." 
    echo ""
}

action=$1

if [ "$action" == "" ]
then
    action="run"
fi

echo ""
pushd ./4-deploy > /dev/null
case "$action" in
    "run")
        ./run.sh
        ;;
    "stop")
        ./stop.sh
        ;;
    "status")
        ./status.sh $2
        ;;
    "logs")
        ./logs.sh $2
        ;;
    "exec")
        ./exec.sh $2
        ;;
    *)
	print_help
        ;;
esac
popd > /dev/null
echo ""
