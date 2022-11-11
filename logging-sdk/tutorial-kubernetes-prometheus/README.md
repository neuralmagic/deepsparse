# DeepSparse + Kubernetes + Prometheus/Grafana

Since DeepSparse runs on CPUs, you can scale deployments elastically with Kubernetes just like any other workload.

This tutorial demonstrates how to monitor a DeepSparse in a Kubernetes cluster with Prometheus.

**There are four steps:**
- Build the DeepSparse Server Docker Image
- Create Kubernetes Cluster + Enable External Communication
- Spin Up the Clients
- Checkout the Info in Grafana

## 0. Setting Up

#### Installation 

This tutotial requires the following:
- Docker
- Minikube

#### Code 

The repository contains all the code you need:

```bash
.
├── client
│   ├── client_image_classification.py
│   ├── client_sentiment_analysis.py
│   └── piglet.jpg
├── demo.sh
├── Dockerfile
├── grafana
│   └── dashboard.json
├── img
├── image_classification_server_config.yaml
├── image_classification_deployment.yaml
├── sentiment_analysis_server_config.yaml
├── sentiment_analysis_deployment.yaml
└── README.md
```

## 1. Build the DeepSparser Server Docker Images

In this tutorial, we will create two model endpoints, one for Image Classification and one for Sentiment Analysis.

We have provided a `Dockerfile`, which downloads DeepSparse and launches a DeepSparse Server with a config file.

We have provided the pre-made `config` files for an image classification and a sentiment analysis Server:
- `image_classification_server_config.yaml`
- `sentiment_analysis_server_config.yaml`

The config files are typical DeepSparse Server configs with logging to Prometheus enabled. Run the following to create a docker image for each task, passing an inference port and proper `config` file.

For the sentiment analysis model (task one):

```bash
docker build -f Dockerfile --build-arg PORT=5543 --build-arg CONFIG='./sentiment_analysis_server_config.yaml' -t sentiment_analysis:latest .
```
For the image classification model (task two):

```bash
docker build -f Dockerfile --build-arg PORT=5544 --build-arg CONFIG='./image_classification_server_config.yaml` -t image_classification:latest .
```

## 2. Create the Kubernetes Cluster

The first step is to create two DeepSparse services with the image classification and sentiment analysis servers that are deployed to the Kubernetes cluster.

First, run the Kuberenetes cluster.

```bash
eval $(minikube docker-env)
minikube delete
minkube start
```

Load the local docker images for boath apps into minikube.

```bash
minikube image load sentiment_analysis:latest
minikube image load image_classification:latest
```

Create the Kubernetes Resources for DeepSparse Server. 
```bash
kubectl apply -f ./sentiment_analysis_deployment.yaml
kubectl apply -f ./image_classification_deployment.yaml
```

The files `sentiment_analysis_deployment.yaml` and `image_classification_deployment.yaml` are declarative methods of create the instances of `Service` and `Deployment` in the cluster.

Let's take a look at `image_classification_deployment.yaml` as an example (ignoring `ServiceMonitor` for a moment). You can see that we will create a Service with port `5544` available as the model service and port `6100` available as the monitoring service.

```yaml
apiVersion: v1
kind: Service   # expose an application running on a set of Pods as a network service
metadata:
  name: image-classification-service
  labels:
    app: image-classification
spec:
  selector:
    app: image-classification # the deployment this app should serve
  ports:
    - name: inference
      port: 5544
      targetPort: 5544        # the exposed port of the model service (for users)
    - name: monitoring
      port: 6100              # the exposed port of the monitoring service (for prometheus)
  clusterIP: 10.101.156.113
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment # create a deployment
metadata:
  name: image-classification
  labels:
    app: image-classification
spec:
  selector:
    matchLabels:
      app: image-classification
  replicas: 2
  template:
    metadata:
      labels:
        app: image-classification
    spec:
      containers: # information about the container we want in the service
        - name: image-classification
          image: image_classification:latest
          imagePullPolicy: IfNotPresent
          ports:
            - name: inference
              containerPort: 5544   # inference port on 5544 (for requests)
            - name: monitoring
              containerPort: 6100   # monitoring port on 6100 (for prometheus)
---
```

Let's confirm Kubernetes set them up properly:

See created pods:
```
kubectl get pods --namespace default
```

See created services:
```
kubectl get svc --namespace default
```

## 3. Add Prometheus Monitoring

### Prometheus Operator

We will use the Prometheus Operator. For more info, see [here](https://blog.container-solutions.com/prometheus-operator-beginners-guide).

Clone and setup the Prometheus Operator using [`kube-prometheus`](https://github.com/prometheus-operator/kube-prometheus)
```bash
git clone https://github.com/prometheus-operator/kube-prometheus.git --depth 1
kubectl create -f kube-prometheus/manifests/setup
kubectl create -f kube-prometheus/manifests/
kubectl get svc --namespace monitoring # displays the services
```
As you can see, server services are now up and running. This includes alert managers and things like Grafana. We care about `prometheus-k8s` and `grafana`.

### ServiceMonitor

The `ServiceMonitor` custom resource defintion allows you to declare how a dynamic set of services should be monitored with desired configuration defined using label selection. Let's take a look at how the `ServiceMonitor` is declared in `image_classification_deployment.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: image-classification-service

# ... same as above
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-classification

# ... same as above
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: image-classification-monitor  # the name of the monitor
  labels:
    app: image-classification
spec:
  selector:
    matchLabels:
      app: image-classification       # the name of the service this monitor should observe
  endpoints:
    - interval: 5s                    # scraping frequency
      port: monitoring
```

We included this `ServiceMonitor` in the `deployment.yaml` files, so it is running within our default namespace.

Run the following to see which `ServiceMonitors` are running:
```bash
kubectl get servicemonitor
```


Enable External Communication with the Cluster
```bash
# create a route to deployed services and sets their Ingress to their ClusterIP
minikube tunnel
```

### Architecture
We now have a cluster with the following architecture.

<img width="50%" src="images/img_1.png"/>

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

