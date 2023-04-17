#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

action=$1

print_help() {
    echo ""
    echo "Usage: $0 <arg>"
    echo ""
    echo "   This script is used for building, deploying, executing, and managing performance and benchmark tests"
    echo "   against model servers and models running on the configured container runtime."
    echo ""
    echo "   Available arguments:"
    echo "   build      - build a test container image"
    echo "   push       - push test image to container registry"
    echo "   pull       - pull test image from container registry if available"
    echo "   run [test] - if a test is not specified, the test container starts as a service in idle mode. "
    echo "                A shell can be opened and tests located in /app/tests or others can be executed manually."
    echo "                If a test is specified, then the test will be run as a job "
    echo "                and the container will exit when the job is complete."
    echo "   exec       - open shell in test container"
    echo "   logs       - show logs of the test container"
    echo "   status     - show status of test container"
    echo "   stop       - stop test container"
    echo ""
    echo "       Available tests:"
    echo "       seq - send a request to each model server and model sequentially"
    echo "       rnd - send random requests to models"
    echo "       bmk - run benchmark test clint to measure throughput and latency under load with random requests"
    echo "       bma - run benchmark analysis - aggregate and average stats from logs of all completed benchmark containers"
    echo "             It is required that bmk completes successfully before bma can produce proper statistics"
    echo ""
    echo "       Example:"
    echo "       $0 run seq"
    echo ""
}

echo ""
case "$action" in
    "build")
        ./5-test/build.sh
        ;;
    "push")
        ./5-test/push.sh
        ;;
    "pull")
        ./5-test/pull.sh
        ;;
    "exec")
        ./5-test/exec.sh $2
        ;;
    "logs")
        ./5-test/logs.sh $2
        ;;
    "status")
        ./5-test/status.sh
        ;;
    "stop")
        ./5-test/stop.sh $2
        ;;
    "run")
        # $2 here is the name of the test to execute, if none specified, then test container will start in sleep mode
        ./5-test/run.sh $2
        ;;
    *)
        print_help
        ;;
esac
echo ""
