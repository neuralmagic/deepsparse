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
- `docker`
- `minikube`
- `kubectl`

#### Code / Setup

In this example, we will deploy two models with DeepSparse Server inside a Kubernetes Cluster. One will be 
a Sentiment Analysis QA Pipeline with BERT and one will be an Image Classification Pipeline with ResNet-50.

Each will be available over HTTP endpoints at ports `5543` and `5544`, and we will use Prometheus
to monitor each service in the cluster.

The repository contains all the code you need:

```bash
.
├── client                                    # Toy Clients For Interacting w/ the Server
│   ├── client_image_classification.py
│   ├── client_sentiment_analysis.py
│   └── piglet.jpg
├── demo.sh
├── Dockerfile                                # Dockerfile for creating DeepSparse Server Images
├── grafana
│   └── dashboard.json                        # Grafana Configuration
├── img
├── image_classification_server_config.yaml   # DeepSparse Server Config
├── image_classification_deployment.yaml      # K8S deployment Config
├── sentiment_analysis_server_config.yaml     # DeepSparse Server Config
├── sentiment_analysis_deployment.yaml        # K8S deployment Config
```

## 1. Build the DeepSparser Server Docker Images

In this tutorial, we will create two model endpoints, one for Image Classification and one for Sentiment Analysis.

We have provided a `Dockerfile`, which downloads DeepSparse and launches a DeepSparse Server with a config file.

We have provided the pre-made `config` files for an image classification and a sentiment analysis Server:
- `image_classification_server_config.yaml`
- `sentiment_analysis_server_config.yaml`

The config files are typical DeepSparse Server configs with logging to Prometheus enabled. Run the following to build a docker image for each task, passing an inference port and proper `config` file.

For the sentiment analysis model (task one):

```bash
docker build -f Dockerfile --build-arg PORT=5543 --build-arg CONFIG='./sentiment_analysis_server_config.yaml' -t sentiment_analysis:latest .
```
For the image classification model (task two):

```bash
docker build -f Dockerfile --build-arg PORT=5544 --build-arg CONFIG='./image_classification_server_config.yaml` -t image_classification:latest .
```

## 2. Create the Kubernetes Cluster with Prometheus

### Start a Cluster

The first step is to start the Kubernetes cluster. We will use [`minikube`](https://minikube.sigs.k8s.io/docs/), which lets you setup a local cluster easily.

```bash
eval $(minikube docker-env)
minikube delete
minkube start
```

### Spin up a Prometheus Operator

We will use the Prometheus Operator, which provided Kubernetes native deployment and management of Prometheus and related monitoring components. For more info, see [here](https://blog.container-solutions.com/prometheus-operator-beginners-guide).

[`kube-prometheus`](https://github.com/prometheus-operator/kube-prometheus) provides example configurations for a complete cluster monitoring stack based on Prometheus and the Prometheus Operator. Clone the repo and create the resources.

```bash
git clone https://github.com/prometheus-operator/kube-prometheus.git --depth 1
kubectl create -f kube-prometheus/manifests/setup
kubectl create -f kube-prometheus/manifests/
kubectl get svc --namespace monitoring # displays the services
```
As you can see, the monitoring services are now up and running. We care about `prometheus-k8s` and `grafana` in this example.

### ServiceMonitor

The `ServiceMonitor` custom resource defintion (CRD) allows you to declare how a dynamic set of services should be monitored with desired configuration defined using label selection. `kube-prometheus` installed this CRD for you.

Run the following to see which `ServiceMonitors` are running:
```bash
kubectl get servicemonitor
```
We will use the `ServiceMonitor` in the deployment configurations for the Model Services below.

## 3. Launch the Model Services

### Load the Model Service Docker Images 
```bash
minikube image load sentiment_analysis:latest
minikube image load image_classification:latest
```
(Note: this may take a minute or two to complete)

### Create the Model Service Resources
```bash
kubectl apply -f ./sentiment_analysis_deployment.yaml
kubectl apply -f ./image_classification_deployment.yaml
```
The files `sentiment_analysis_deployment.yaml` and `image_classification_deployment.yaml` are declarative methods of create the instances of `Service` and `Deployment` in the cluster.

Let's take a look at `image_classification_deployment.yaml` as an example. You can see that we  create a Service with port `5544` available as the model service and port `6100` available as the monitoring service. Additionally, we have declared a `ServiceMonitor`, which automatically generates a Prometheus scrape configuration.

<details>
  <summary>Click to view</summary>
  
```yaml
# image_classification_deployment.yaml

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
  
</details>

See created pods:
```
kubectl get pods --namespace default
```

See created services:
```
kubectl get svc --namespace default
```

### Enable External Communication with the Cluster
Open up a new terminal and run the following:

Note that the `tunnel` does not return. You must keep this process running.

```bash
# create a route to deployed services and sets their Ingress to their ClusterIP
minikube tunnel --cleanup

>> Status:	
>>	machine: minikube
>>	pid: 53825
>>	route: 10.96.0.0/12 -> 192.168.49.2
>>	minikube: Running
>>	services: [image-classification-service, sentiment-analysis-service]
>>    errors: 
>>		minikube: no errors
>>		router: no errors
>>		loadbalancer emulator: no errors
```

### Final Architecture
We now have a cluster with the following architecture:

<img width="75%" src="img/img_1.png"/>

## 3. Spin up the Clients
We created two toy clients that send a request to the model services once every 5 seconds and prints the response to the command line.

Every client script takes three arguments (in the order):
- Input to the Engine (text or image)
- Inference Port Number
- Service IP Number

```bash
python client/client_sentiment_analysis.py "this is a really cute piglet!" 5543 10.101.156.112
python client/client_image_classification.py piglet.jpg 5544 10.101.156.113
```

## 4. Plot in Grafana

[**TO BE UPDATED** Once We Have Metrics Working]

```bash
# (optionally) expose the port to validate on localhost: 9090 that everything has been properly setup in Prometheus
kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090
```
![img.png](img/img.png)
```bash
# expose the port 3000 to interact with the Grafana on localhost:3000
# note: 
kubectl --namespace monitoring port-forward svc/grafana 3000
```
![img_5.png](img/img_2.png)

Note: When setting up Prometheus data source in Grafana, we need to either:
- manually change it to: http://prometheus-k8s.monitoring.svc:9090 
- configure it programmatically in manifest files (holding off this decision, since I am not sure whether this is caused by ssh tunneling or not).

