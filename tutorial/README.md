
DeepSparse Server First, let’s have some context on the monitoring stack that we will be using during this guide. 
To leverage the metrics that Jina exposes, we recommend using the Prometheus/Grafana stack. 
In this setup, Jina will expose different metrics endpoint, and Prometheus will then be in charge of scraping these endpoints, as well as collecting, aggregating, and storing the different metrics. Prometheus will then allow external entities (like Grafana) to access these aggregated metrics via the query language PromQL. Then the role of Grafana here will be to allow users to visualize these metrics by creating dashboards.

This tutorial will show you how to monitor the DeepSparse server using Prometheus/Grafana stack.

The repository has the following structure:

```bash
.
├── client 
│   ├── client.py # simple client application
│   └── piglet.jpg 
├── deepsparse_server_config.yaml # specifies the configuration of the DeepSparse server
├── demo.sh # script that runs the contents of this tutorial
├── docker # specifies the configuration of the containerized Prometheus/Grafana stack
│   ├── docker-compose.yaml
│   └── prometheus.yaml
└── grafana # specifies the design of the Grafana dashboard
    └── dashboard.json
```

All the steps described in this tutorial are summarized in the `demo.sh` script.


## Spin up the DeepSparse Server

The file `deepsparse_server_config.yaml` specifies the configuration of the server. Once we launch the server, it creates a sample `image_classification` pipeline that runs the sparsified model from SparseZoo. The server also exposes two endpoints:

- port `6100`: exposes the `metrics` endpoint through [Prometheus python client](https://github.com/prometheus/client_python). This is the endpoint that the Prometheus service is to scrape for logs.
- port `5543`: exposes the endpoint for inference.

To spin up the server:
```
pip install deepsparse[server]
deepsparse.server config deepsparse_server_config.yaml
```

To validate, that metrics are being properly exposed, visit `localhost:6100`. It should contain logs in the specific format meant to be used by the PromQL query language.
## Setup Prometheus/Grafana Stack

To start up a Prometheus stack to monitor the DeepSparse server, run:

```bash
cd docker
docker-compose up
```

Note that inside the file `docker/prometheus.yaml`, the `scrape_config` has the information about the `metrics` endpoint exposed by the server on port `6100`.

## Launch the Python Client and Run Inference Continuously

Run:

```bash
python client/client.py client/piglet.jpg 5543
```

to instantiate a simple client, that periodically sends requests to the server. 

Note: the first argument is the path to the sample image, while the second argument is the port number that matches the inference endpoint of the server.

## Inspecting the Prometheus/Grafana Stack

You may visit `localhost:9090` to inspect whether Prometheus recognizes the `metrics` endpoint (`Status` -> `Targets`)

![img.png](images/img_1.png)

Visit `localhost:3000` to launch Grafana. Log in with the default username (`admin`) and password (`admin`). Setup the Prometheus data source (`Add your first data source` -> `Prometheus`). Now you should be ready to create/import your dashboard. If you decide to import the dashboard, either upload `grafana/dashboard.json` or paste its contents using Grafanas `import` functionality.

![img.png](images/img_2.png)



