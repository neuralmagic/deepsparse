#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

print_help() {
	echo ""
	echo "Usage: $0"
	echo ""
	echo "   This script just opens the global configuration file (config.properties) in a text editor."
	echo "   By default we use vi, but this can be easily changed by modifying the script."
	echo "   Changes to the config file take effect with the next action script execution."
	echo ""
}

if [ "$1" == "" ]; then
	vi ./config.properties
else
	print_help
fi
