# DeepSparse Pipeline + Prometheus/Grafana

DeepSparse Logging is compatible with Prometheus/Grafana, making it easy to stand up a model monitoring service.

This tutorial will show you how to connect DeepSparse Logging from a Pipeline to the Prometheus/Grafana stack.

#### There are four steps:
- Create a Logging Config
- Point Prometheus to the appropriate endpoint to scape the data at a specified interval
- Run a DeepSparse Pipeline with Logging Configured
- Visualize data in Prometheus with dashboarding tool like Grafana

## 0. Setting Up
#### Installation

To run this tutorial, you need Docker, Docker Compose, and DeepSparse Server
- [Docker Installation](https://docs.docker.com/engine/install/)
- [Docker Compose Installation](https://docs.docker.com/compose/install/)
- DeepSparse Server is installed via PyPI (`pip install deepsparse`)

#### Code
The repository contains all the code you need:

```bash
.
├── run_inference.py
├── config.yaml
├── piglet.jpeg
├── docker                  # specifies the configuration of the containerized Prometheus/Grafana stack
│   ├── docker-compose.yaml
│   └── prometheus.yaml
└── grafana                 # specifies the design of the Grafana dashboard
    └── dashboard.json
```
## 1. Create a Logging Config

`config.yaml` will be used by the `ManagerLogger` to configure DeepSparse Logging.

```yaml
# config.yaml

loggers:                      
  prometheus:           
    port: 6100
    
# all system logs on (all are pre-defined)
system_logging: on

# specify which data logs to perform
data_logging:
    pipeline_inputs:
    # to be updated once we have code landed
```

The config file instructs sets up Prometheus to logs on port `6100`, turns on all system logging, and sets up
the following data logs:
- [To be updated once we have code landed]

Thus, when inference is run from the Pipeline, port `6100` exposes the `metrics` endpoint through [Prometheus python client](https://github.com/prometheus/client_python).

## 2. Setup Prometheus/Grafana Stack

For simplicity, we have provided `docker-compose.yaml` that spins up the containerized Prometheus/Grafana stack. 

In that file, we instruct `prometheus.yaml` (a [Prometheus config file](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)) to be passed to the Prometheus container. Inside `prometheus.yaml`, the `scrape_config` has the information about the `metrics` endpoint exposed by the server on port `6100`.

<details>
    <summary>Click to see Docker Compose File</summary>

```yaml    
# docker-compose.yaml
    
version: "3"

services:
  prometheus:
    image: prom/prometheus
    extra_hosts:
      - "host.docker.internal:host-gateway"     # allow a direct connection from container to the local machine
    ports:
      - "9090:9090" # the default port used by Prometheus
    volumes:
      - ${PWD}/prometheus.yaml:/etc/prometheus/prometheus.yml # mount Prometheus config file

  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    ports:
      - "3000:3000" # the default port used by Grafana

```
</details>

<details>
    <summary>Click to see Prometheus Config File</summary>
    
```yaml
# prometheus.yaml
    
global:
  scrape_interval: 15s                      # how often to scrape from endpoint
  evaluation_interval: 30s                  # time between each evaluation of Prometheus' alerting rules

scrape_configs:
  - job_name: prometheus_logs               # your project name
    static_configs:
      - targets:
          - 'host.docker.internal:6100'     # should match the port exposed by the PrometheusLogger in the DeepSparse Server config file 
```
</details>

To start up a Prometheus stack to monitor the Pipeline, run:

```bash
cd docker
docker-compose up
```

## 3. Run Inference

`run_inference.py` is a simple script that runs an Image Classification Pipeline repeatedly. Importantly,
the Pipeline uses a Logger with the configuration from step 1, so logging metrics will be exposed on `6100`.

``` python
# run_inference.py

from deepsparse import Pipeline
from time import sleep

# SparseZoo model stub or path to ONNX file
model_path = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none"

# logger object referencing the local logging config file
logger = ManagerLogger(config="config.yaml")

# pipeline instantiated with the config file
img_classification_pipeline = Pipeline.create(
    task="image-classification",
    model_path=model_path,
    logger=logger
)

# runs for 2 minutes 
for _ in range(120):
    preds = img_classification_pipeline(["piglet.jpg"])
    print(preds)
    sleep(1)
```

Run it with the following:

```bash
python run_inference.py
```
The script will continously run inference on `piglet.jpg` once per second for 2 minutes.

To validate that metrics are being properly exposed, visit `localhost:6100`. It should contain logs in the specific format meant to be used by the PromQL query language.

## 4. Inspecting the Prometheus/Grafana Stack

You may visit `localhost:9090` to inspect whether Prometheus recognizes the `metrics` endpoint (`Status` -> `Targets`)

![img.png](images/img_1.png)

Visit `localhost:3000` to launch Grafana. Log in with the default username (`admin`) and password (`admin`). 
Setup the Prometheus data source (`Add your first data source` -> `Prometheus`). 

Now you should be ready to create/import your dashboard. If you decide to import the dashboard, either upload `grafana/dashboard.json` or 
paste its contents using Grafana's `import` functionality.

![img.png](images/img_2.png)
