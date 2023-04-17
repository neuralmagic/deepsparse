#!/bin/bash

# This script installs Cloud9

# Install dev tools
sudo yum -y groupinstall "Development Tools"

# Install nodejs
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.39.0/install.sh | bash
source ~/.nvm/nvm.sh
source ~/.bashrc
nvm install 16.15.1

# Install Cloud9
curl -L https://raw.githubusercontent.com/c9/install/master/install.sh | bash

