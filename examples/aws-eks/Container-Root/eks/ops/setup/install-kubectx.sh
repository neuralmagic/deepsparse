#!/bin/bash

pushd /tmp
git clone https://github.com/ahmetb/kubectx
mv kubectx /opt
ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
ln -s /opt/kubectx/kubens /usr/local/bin/kubens
popd

