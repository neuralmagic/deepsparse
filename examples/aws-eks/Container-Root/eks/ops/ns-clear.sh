#!/bin/bash

#This script clear any namespaces that are stuck in the Terminating state

mapfile -t namespaces < <( kubectl get namespace | grep Terminating | awk '{print $1}' )

if [ "${#namespaces[@]}" == "0" ]; then
	echo "No Terminating namespaces found"
else
	echo "Clearing namespaces that are in Terminating state ..."

	for namespace in "${namespaces[@]}"
	do
    		echo "Clearing namespace $namespace ..."
    		kubectl get namespace "$namespace" -o json > "$namespace.json"
    		tmpfile=$(mktemp)
    		echo $(jq '.spec.finalizers |= []' "$namespace.json") | jq '.' > "$tmpfile"
    		cp "$tmpfile" "$namespace.json"
    		rm $tmpfile
    		curl -k -H "Content-Type: application/json" -X PUT --data-binary @$namespace.json http://127.0.0.1:8001/api/v1/namespaces/$namespace/finalize
    		rm "$namespace.json"
	done
fi

