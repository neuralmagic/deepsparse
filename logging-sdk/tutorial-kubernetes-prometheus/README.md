# DeepSparse + Kubernetes + Prometheus/Grafana

Since DeepSparse is CPU-only, you can scale deployments elastically with Kubernetes just like any other workload.

DeepSparse is also integrated with Prometheus, which simplifies the process of standing up a model monitoring service.

This tutorial demonstrates how to monitor a DeepSparse in a Kubernetes cluster.

**There are five steps:**
- Build the DeepSparse Server Docker Image
- Create Kubernetes Cluster + Enable External Communication
- Spin Up the Clients
- Checkout the Info in Grafana

## 0. Setting Up

#### Installation 

This tutotial requires the following:
- Docker
- Docker Compose
- Minikube
- DeepSparse Server (`pip install deepsparse[server]`)

#### Code 

The repository contains all the code you need:

```bash
.
├── client
│   ├── client_image_classification.py
│   ├── client_sentiment_analysis.py
│   └── piglet.jpg
├── demo.sh
├── docker
│   └── Dockerfile
├── grafana
│   └── dashboard.json
├── images
├── kubernetes
│   ├── image_classification
│   │   ├── config.yaml
│   │   └── deployment.yaml
│   └── sentiment_analysis
│       ├── config.yaml
│       └── deployment.yaml
└── README.md
```

## 1. Build the DeepSparser Server Docker Images

In this tutorial, we will create two model endpoints, one for Image Classification and one for Sentiment Analysis.

Each endpoint has a DeepSparse Server config file found in `/kubernetes/image_classification/config.yaml` and `kubernetes/sentiment_analysis/config.yaml`. They are the typical DeepSparse Server config files with logging to Prometheus enabled.

Run the following to create a docker image for each task, passing the inference port (where the endpoint will be exposed) and copying over the proper `config.yaml` file.

For the sentiment analysis model (task one):

```bash
docker build -f /home/.../docker/Dockerfile --build-arg PORT=5543 --build-arg CONFIG='home/.../kubernetes/sentiment_analysis/config.yaml -t sentiment_analysis:latest .
```
For the image classification model (task two):

```bash
docker build -f /home/.../docker/Dockerfile --build-arg PORT=5544 --build-arg CONFIG='home/.../kubernetes/image_classification/config.yaml -t image_classification:latest .
```

## 2. Create the Kubernetes Cluster

### Architecture
We will create a cluster with the following architecture:

<img width="50%" src="images/img_1.png"/>

Enable the use of local docker images:
```bash
eval $(minikube docker-env)
```

Launch the minikube Kubernetes cluster:
```bash
minkube start
```

Make the `sentiment_analysis` and `image_classification` images available in the cluster
```bash
minikube image load sentiment_analysis:latest
minikube image load image_classification:latest
```

Clone and setup the Prometheus Operator using [`kube-prometheus`](https://github.com/prometheus-operator/kube-prometheus)
```bash
git clone https://github.com/prometheus-operator/kube-prometheus.git --depth 1
kubectl create -f kube-prometheus/manifests/setup
kubectl create -f kube-prometheus/manifests/
```

Create the Kubernetes Resources for DeepSparse Server
```bash
kubectl apply -f /home/.../kubernetes/sentiment_analysis/deployment.yaml
kubectl apply -f /home/.../kubernetes/image_classification/deployment.yaml
```

Enable External Communication with the Cluster
```bash
# create a route to deployed services and sets their Ingress to their ClusterIP
minikube tunnel
```

## 3. Spin up the Clients
Every client script takes three arguments (in the order):
- input to the engine (text or image)
- inference port number
- service ip number

```bash
python client/client_sentiment_analysis.py "this is a really cute piglet!" 5543 10.101.156.112
python client/client_image_classification.py piglet.jpg 5544 10.101.156.113
```

## 4. Plot in Grafana

To be continued

```bash
# (optionally) expose the port to validate on localhost: 9090 that everything has been properly setup in Prometheus
kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090
```
![img.png](images/img.png)
```bash
# expose the port 3000 to interact with the Grafana on localhost:3000
# note: 
kubectl --namespace monitoring port-forward svc/grafana 3000
```
![img_5.png](images/img_2.png)

Note: When setting up Prometheus data source in Grafana, we need to either:
- manually change it to: http://prometheus-k8s.monitoring.svc:9090 
- configure it programmatically in manifest files (holding off this decision, since I am not sure whether this is caused by ssh tunneling or not).

