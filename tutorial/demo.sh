#!/bin/bash

virtual_env_path="..."

docker_files_path="${PWD}/docker"
client_path="${PWD}/client"

server_config_file_path="${PWD}/deepsparse_server_config.yaml"
sample_image_path="${client_path}""/piglet.jpg"
port="5543" 

tmux new-session \; \
send-keys 'cd '${docker_files_path}'' C-m \; \
send-keys 'docker-compose up' C-m \; \
split-window -h \; \
send-keys '. '${virtual_env_path}'/bin/activate' C-m \; \
send-keys 'deepsparse.server config deepsparse_server_config.yaml' C-m \; \
split-window -h \; \
send-keys 'sleep 10' C-m \; \
send-keys 'python '${client_path}'/client.py '${sample_image_path}' '${port}'' C-m \; \
