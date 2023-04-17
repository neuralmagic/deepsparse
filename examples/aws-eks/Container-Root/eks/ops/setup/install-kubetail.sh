#!/bin/bash


curl -o /tmp/kubetail https://raw.githubusercontent.com/johanhaleby/kubetail/master/kubetail
chmod +x /tmp/kubetail
mv /tmp/kubetail /usr/local/bin/kubetail

