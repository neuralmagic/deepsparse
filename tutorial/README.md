# Prometheus Operator in Kubernetes

## Architecture
![img_4.png](images/img_1.png)
## Project Structure

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

## Step-by-step execution

### 1. Build the DeepSparser Server Docker Images

We prepare the docker image for every task by passing the inference port and copying over the proper `config.yaml` file.

For the sentiment analysis model (task one):

```bash
docker build -f /home/.../docker/Dockerfile --build-arg PORT=5543 --build-arg CONFIG='home/.../kubernetes/sentiment_analysis/config.yaml -t sentiment_analysis:latest .
```
For the image classification model (task two):

```bash
docker build -f /home/.../docker/Dockerfile --build-arg PORT=5544 --build-arg CONFIG='home/.../kubernetes/image_classification/config.yaml -t image_classification:latest .
```
Now both images are built locally.

### 2. Setup the Kubernetes Cluster

```bash
# enable the use of local docker for minikube - to be able to list in minikube the docker images available locally
eval $(minikube docker-env)
# launch the minikube Kubernetes cluster
minikube start	
# makes sentiment_analysis image available in the cluster
minikube image load sentiment_analysis:latest
# makes image_classification image available in the cluster
minikube image load image_classification:latest

# clone and setup the Prometheus Operator using kube-prometheus (https://github.com/prometheus-operator/kube-prometheus)
git clone https://github.com/prometheus-operator/kube-prometheus.git --depth 1
kubectl create -f kube-prometheus/manifests/setup
kubectl create -f kube-prometheus/manifests/
```

### 3. Create Kubernetes Resources for Deepsparse Servers

```bash
kubectl apply -f /home/.../kubernetes/sentiment_analysis/deployment.yaml
kubectl apply -f /home/.../kubernetes/image_classification/deployment.yaml
```

### 4. Enable External Communication with the Cluster
```bash
# create a route to deployed services and sets their Ingress to their ClusterIP
minikube tunnel
```

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

### 5. Spin up the Clients
Every client script takes three arguments (in the order):
- input to the engine (text or image)
- inference port number
- service ip number

```bash
python client/client_sentiment_analysis.py "this is a really cute piglet!" 5543 10.101.156.112
python client/client_image_classification.py piglet.jpg 5544 10.101.156.113
```

### 6. Plot in Grafana

To be continued

