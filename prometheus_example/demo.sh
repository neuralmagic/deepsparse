#!/usr/bin/env bash
app="./app"
app2="./app2"
# start kubernetes cluster using minikube
tmux new-session \; \
send-keys 'eval $(minikube docker-env)' C-m \; \
send-keys 'minikube delete' C-m \; \
send-keys 'minikube start' C-m \; \
send-keys 'minikube image load hello-python:latest' C-m \; \
send-keys 'minikube image load hello-python_2:latest' C-m \; \
send-keys 'git clone https://github.com/prometheus-operator/kube-prometheus.git --depth 1' C-m \; \
send-keys 'kubectl create -f kube-prometheus/manifests/setup' C-m \; \
send-keys 'kubectl create -f kube-prometheus/manifests/' C-m \; \
send-keys 'sleep 70' C-m \; \
send-keys 'kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090' C-m \; \
split-window -h \; \
send-keys 'cd '$app'' C-m \; \
send-keys 'docker build --no-cache -f Dockerfile -t hello-python:latest .' C-m \; \
send-keys 'cd ..' C-m \; \
send-keys 'cd '$app2'' C-m \; \
send-keys 'docker build --no-cache -f Dockerfile -t hello-python_2:latest .' C-m \; \
split-window -v \; \
send-keys 'sleep 70' C-m \; \
send-keys 'kubectl apply -f deployment.yaml' C-m \; \
send-keys 'kubectl apply -f deployment2.yaml' C-m \; \
split-window -v \; \
send-keys 'sleep 100' C-m \; \
send-keys 'kubectl --namespace monitoring port-forward svc/grafana 3000' C-m \; \
