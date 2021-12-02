#!/bin/bash

# Integration setup command to setup the folder so it is ready to run YOLACT models for inference.
# Creates a yolact folder next to this script with all required dependencies from the dbolya/yolact repository.
# Command: `bash setup_integration.sh`

git clone https://github.com/neuralmagic/yolact.git
cd yolact
git checkout release/0.9
pip install -r requirements.txt
pip install deepsparse
