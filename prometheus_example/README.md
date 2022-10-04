# An Example Of How To Deploy Prometheus Operator

The Prometheus Operator is a service that manages Prometheus (and related monitoring components) clusters atop Kubernetes. It simplifies and automates the configuration of a Prometheus-based monitoring stack for the Kubernetes cluster.


## Walkthrough

Note: To automate the whole process described in this readme I created one single master script `demo.sh`. All the contents of the writeup can be launched by a simple

```bash
bash demo.sh
```

### Simulate Kubernetes Cluster To Be Monitored

The first step is to simulate two `DeepSparse` services that are deployed to the Kubernetes cluster. The great thing about Prometheus Operator is that it can easily figure out on its own how many apps to track.

Let's take a look at the simple service at `/apps/main.py`. It continuously logs a random number as a metric called `test_python_metric` and exposes it to the endpoint on the port `8000`. The second service, located in `/apps2/` works analogously - here the metric is called `test_python_metric_2`. 

Note: It is not necessary to define two apps. I could simply deploy the same `/app/main.py` as two different Kubernetes services, and I would be able to tell which one is which by the service name label. Also, I do not need to worry about port management. As long as the port `8000` (or any other port that we assign to the python client) is free in the context of the single service and consistently declared, everything will be fine.

#### Deploy the App to Kubernetes

```bash
# run the Kubernetes cluster 
# also make sure that there are no dangling minikube instances

eval $(minikube docker-env) # set docker env (unix shell)
minikube delete
minikube start

# locally build docker images for both apps and load them directly to minikube

cd /app/
docker build --no-cache -f Dockerfile -t hello-python:latest .
minikube image load hello-python:latest

cd ...

cd /app2/
docker build --no-cache -f Dockerfile -t hello-python_2:latest .
minikube image load hello-python_2:latest

# deploy the apps
kubectl apply -f deployment.yaml
kubectl apply -f deployment2.yaml
``` 
Files `deployment.yaml` and `deployment2.yaml` are declarative methods of creating instances of interest (`Deployment` and `Service`) in the cluster. 

Let's take a look at `deployment.yaml` (I will be omitting the declaration of `ServiceMonitor` for now):

```bash
apiVersion: apps/v1
kind: Deployment # describes the Deployment
metadata:
  name: hello-python # deployment name
  labels:
    app: hello-python # adds `app` label to the deployment
spec: # how the Deployment finds which Pods to manage
  selector:
    matchLabels:
      app: hello-python
  replicas: 2
  template:
    metadata:
      labels:
        app: hello-python
    spec:
      containers: # the information about the thing (image) that we want to containerize 
      - name: hello-python-container
        image: hello-python:latest
        imagePullPolicy: Never
        ports:
        - name: web
          containerPort: 8000 # port to expose from the container
---
apiVersion: v1
kind: Service # expose an application running on a set of Pods as a network service
metadata:
  name: hello-python
  labels:
    app: hello-python
spec:
  selector:
    app: hello-python # the name of the deployment this app should serve
  ports:
  - name: web
    port: 8000 # the exposed port of the service 
  type: LoadBalancer
```

After we have created both deployments, we can conclude that Kubernetes set them up properly:

See created pods: 

```bash 
kubectl get pods --namespace default
```
<img width="493" alt="image" src="https://user-images.githubusercontent.com/97082108/193821258-b3444832-8765-4185-b5f9-67471016192e.png">

See created services:

```bash 
kubectl get svc --namespace default
```
<img width="593" alt="image" src="https://user-images.githubusercontent.com/97082108/193821333-8ff38db9-5cea-4bb5-9ba8-7bcf377a0745.png">


Let's expose the endpoint to one of the services to assert, that it is exposing the expected metrics properly:

```bash
kubectl port-forward svc/hello-python 8000
```
then check your `http://localhost:8000` 
<img width="645" alt="image" src="https://user-images.githubusercontent.com/97082108/193821477-0d9990b1-a37f-4549-a55c-c4a011a6e2f4.png">


Note: you may need to do SHH tunneling if working on the remote machine: 

```bash
ssh -N -f -L localhost:8000:localhost:8000 <username>@<address>
```

### Spin up the Prometheus Operator 
In the context of Kubernetes, operators are something of a "software extension". They provide a consistent approach to handling all the application operational processes automatically, without any human intervention, which they achieve through close cooperation with the Kubernetes API. For more information on the Prometheus Operator, I'd suggest taking a look [here](https://blog.container-solutions.com/prometheus-operator-beginners-guide). 

There are two ways to install the operator - either through [kube-prometheus stack](https://github.com/prometheus-operator/kube-prometheus) or [helm chart](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack). As far as I know, [both methods do the same thing/achieve the same result/can be quickly swapped](https://stackoverflow.com/questions/54422566/what-is-the-difference-between-the-core-os-projects-kube-prometheus-and-promethe). I choose to use `kube-prometheus`. The installation is simple:

```bash
git clone https://github.com/prometheus-operator/kube-prometheus.git --depth 1 # clone the repo
kubectl create -f kube-prometheus/manifests/setup
kubectl create -f kube-prometheus/manifests/
kubectl get svc --namespace monitoring # display the services spun up by the operator
```
<img width="746" alt="image" src="https://user-images.githubusercontent.com/97082108/193821575-0a1caf24-97a9-4864-8bc4-285e8e766356.png">


As you can see, several services are now up and running. This includes alert managers, but also Grafana. We can potentially explore these services further. However, in the context of this writeup we care about two services: `prometheus-k8s` and `grafana`.

#### Adding a `ServiceMonitor` to your service.
The `ServiceMonitor` CRD (custom resource definition) allows to declaratively define how a dynamic set of services should be monitored. Which services are selected to be monitored with the desired configuration is defined using label selections. This allows an organization to introduce conventions around how metrics are exposed and then following these conventions new services are automatically discovered, without the need to reconfigure the system. In my toy example, I will attach a service monitor to every deployed app. Let's take a look at how the service monitor is declared in `deployment.yaml`:

```bash
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: hello-python # name of the monitor
  labels:
    app: hello-python 
spec:
  selector:
    matchLabels:
      app: hello-python # the name of the service this monitor should be observing
  endpoints:
  - interval: 1s # scraping frequency
    port: web
```

We might now reapply `kubectl apply -f deployment.yaml` (as well as `deployment2.yaml`) and observe which service monitors are there within our default namespace.

```bash
kubectl get servicemonitor
kubectl describe servicemonitor/hello-python
```
<img width="360" alt="image" src="https://user-images.githubusercontent.com/97082108/193821678-eca600b9-129d-43d6-bccc-ca0517384091.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/97082108/193821717-a34069d2-5308-40c6-8963-2ff7ed58e581.png">


### Discover the metrics in Prometheus and Grafana. 
Now, we might take a look at both Prometheus and Grafana and see whether our app metrics are being scraped. First, we need to do some port forwarding (also not forget SSH tunneling):

```bash
kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090
```
<img width="586" alt="image" src="https://user-images.githubusercontent.com/97082108/193822019-3277d537-c7ef-4e9d-b5cb-f43e4ac9e869.png">
<img width="2534" alt="image" src="https://user-images.githubusercontent.com/97082108/193822207-258f0108-2970-413e-a56a-d4877bd2879b.png">


Looking good

```bash
kubectl --namespace monitoring port-forward svc/grafana 3000
```
Note: For some reason the default URL (`http;//localhost:9090` on my machine) of the Prometheus is wrong, this may be due to SSH tunneling. I had to locally change it to:
`http://prometheus-k8s.monitoring.svc:9090` (can be manually configured in manifest files).
<img width="586" alt="image" src="https://user-images.githubusercontent.com/97082108/193821827-bd2b72ec-80f9-4592-ad53-9d611b0b9281.png">
<img width="2202" alt="image" src="https://user-images.githubusercontent.com/97082108/193822276-328b5345-7ad5-43f3-ad7a-d1fef2f6a4a5.png">
Looking good!

Note: We have four graphs -> two apps, each containing two replicas of itself.



## QnA
**Question:** What other interesting features should we explore?

**Answer:** It would be cool to learn more about the alert managers. Would be also great to instruct the users on how to set up alerts when some custom, engine-related metrics get out of hand. Also, let's not forget that the Operator gives us a ton of metrics about our services off-the-shelf - it is obviously monitoring all the cluster, not only the custom metrics. We should absolutely explore those as well and pick the ones that our customers will be interested in.

**Question:** What about making the logs persistent?

**Answer:** Prometheus-Operator: ["Kubernetes supports several kinds of storage volumes. The Prometheus Operator works with PersistentVolumeClaims, which support the underlying PersistentVolume to be provisioned when requested."](https://github.com/prometheus-operator/prometheus-operator/blob/main/Documentation/user-guides/storage.md)

**Question:** What about the dashboards? Can we set them up in advance for the users and use the default configuration in the Operator's Grafana?

**Answer:** [Not trivial, but doable](https://stackoverflow.com/questions/57322022/stable-prometheus-operator-adding-persistent-grafana-dashboards). JinaAI does something [similar](https://docs.jina.ai/how-to/monitoring/).


## Useful sources

Some were pretty hard to find, want to keep them for the future:

- https://prometheus-operator.dev/docs/operator/design/
- https://github.com/prometheus-operator/prometheus-operator/blob/main/Documentation/design.md#servicemonitor
- https://docs.jina.ai/how-to/monitoring/
- https://stackoverflow.com/questions/64319806/kubernetes-servicemonitor-added-but-no-targets-discovered-0-0-up
- https://blog.pilosus.org/posts/2019/06/01/prometheus-operator-no-active-targets/
- https://fabianlee.org/2022/07/07/prometheus-monitoring-a-custom-service-using-servicemonitor-and-prometheusrule/
- https://github.com/prometheus-operator/prometheus-operator/blob/main/Documentation/additional-scrape-config.md
- https://www.cncf.io/blog/2021/10/25/prometheus-definitive-guide-part-iii-prometheus-operator/
- https://www.youtube.com/watch?v=YDtuwlNTzRc&t
- https://github.com/delvin1933/flask-prometheus-ex
- https://dev.to/camptocamp-ops/integrate-an-application-with-prometheus-operator-and-package-with-a-helm-chart-1159
- https://sysdig.com/blog/kubernetes-monitoring-prometheus-operator-part3/
- https://github.com/prometheus-operator/prometheus-operator/issues/2515
- https://github.com/prometheus-operator/kube-prometheus/issues/473









