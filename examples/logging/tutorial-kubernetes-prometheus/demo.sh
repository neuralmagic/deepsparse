#!/bin/bash

virtual_env_path="/home/damian/deepsparse_venv" # e.g. ".../deepsparse_venv"

kubernetes_files_path="kubernetes"
dockerfile_path="${PWD}/docker/Dockerfile"
client_path="${PWD}/client"

server_config_file_path="${PWD}/deepsparse_server_config.yaml"

task_1="sentiment_analysis"
task_2="image_classification"
sample_image_path="${client_path}""/piglet.jpg"
sample_text_input="this is a really cute piglet!"
port_1="5543"
port_2="5544" 
ip_1="10.101.156.112"
ip_2="10.101.156.113"

tmux new-session \; \
send-keys 'rm -rf kube-prometheus' C-m \; \
send-keys 'eval $(minikube docker-env)' C-m \; \
send-keys 'minikube delete' C-m \; \
send-keys 'minikube start' C-m \; \
send-keys 'minikube image load '${task_1}':latest' C-m \; \
send-keys 'minikube image load '${task_2}':latest' C-m \; \
send-keys 'git clone https://github.com/prometheus-operator/kube-prometheus.git --depth 1' C-m \; \
send-keys 'kubectl create -f kube-prometheus/manifests/setup' C-m \; \
send-keys 'kubectl create -f kube-prometheus/manifests/' C-m \; \
send-keys 'sleep 100' C-m \; \
send-keys 'kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090' C-m \; \
split-window -h \; \
send-keys 'docker build -f '${dockerfile_path}' --build-arg PORT='${port_1}' --build-arg CONFIG='${kubernetes_files_path}'/'${task_1}'/config.yaml -t '${task_1}':latest .' C-m \; \
send-keys 'sleep 240' C-m \; \
send-keys 'kubectl apply -f '${kubernetes_files_path}'/'${task_1}'/deployment.yaml' C-m \; \
send-keys 'sleep 300' C-m \; \
send-keys 'python '${client_path}'/client_'${task_1}'.py '${sample_text_input}' '${port_1}' '${ip_1}'' C-m \; \
split-window -h \; \
send-keys 'sleep 300' C-m \; \
send-keys 'kubectl --namespace monitoring port-forward svc/grafana 3000' C-m \; \
split-window -h \; \
send-keys 'docker build -f '${dockerfile_path}' --build-arg PORT='${port_2}' --build-arg CONFIG='${kubernetes_files_path}'/'${task_2}'/config.yaml -t '${task_2}':latest .' C-m \; \
send-keys 'sleep 240' C-m \; \
send-keys 'kubectl apply -f '${kubernetes_files_path}'/'${task_2}'/deployment.yaml' C-m \; \
send-keys 'sleep 300' C-m \; \
send-keys 'python '${client_path}'/client_'${task_2}'.py '${sample_image_path}' '${port_2}' '${ip_2}'' C-m \; \
split-window -v \; \
send-keys 'sleep 250' C-m \; \
send-keys 'minikube tunnel' C-m \; \
