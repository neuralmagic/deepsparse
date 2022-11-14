# DeepSparse + Milvus: Text-Search with BERT

This example demonstrates how to create a semantic search engine using FastAPI, DeepSparse, Milvus, and MySQL.

We will create 4 services:
- Milvus Server - vector database used to hold the embeddings of the article dataset and perform the search queries
- MySQL Server - holds the mapping from Milvus ids to original article data
- DeepSparse Server - inference runtime used to generate the embeddings for the queries
- Application Server - endpoint called by the client side with queries for searching

We will demonstrate running on a local machine as well as in a VPC on AWS with independent-scaling of the App, Database, and Model Serving Components.

## Application Architecture

We have provided a sample dataset in `client/example.csv`. These data are articles about various topics, in `(title,text)` pairs. We will create an application that will allow users to upload arbitrary `text` and find the 10 most similiar articles using semantic search.

The app server is built on FastAPI and exposes a both `/load` and `/search` endpoints. 

The `/load` endpoint accepts a csv file with `(title, text)` representing a series of articles. On `/load`, we project the `text` into the embedding space with BERT running on DeepSparse. We then store each embedding in Milvus with a primary key `id` and store the `(id,title,text)` tripes in MySQL.

The `/search` endpoint enables clients to send `text` to the server. The app server sends the `text` to DeepSparse Server, which returns the embedding of the query. This embedding is sent to Milvus, which searches for the 10 most similiar vectors in the database and returns their `ids` to the app server. The app server then looks up the `(title,text)` in MySQL and returns them back to the client.

As such, we can scale the app server, databases, and model service independently!

## Running Locally

### Start the Server

#### Installation:
- Milvus and Postgres are installed using Docker containers. [Install Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/linux/).
- DeepSparse is installed via PyPI. Create a virtual enviornment and run `pip install -r server/deepsparse-requirements.txt`.
- The App Server is based on FastAPI. Create a virtual enviornment and run `pip install -r server/app-requirements.txt`.

#### 1. Start Milvus

Milvus has a convient `docker-compose` file which can be downloaded with `wget` that launches the necessary services needed for Milvus. 

``` bash
cd server/database-server
wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml -O docker-compose.yml
sudo docker-compose up
cd ..

```
This command should create `milvus-etcd`, `milvus-minio`, and `milvus-standalone`.

#### 2. Start MySQL

MySQL can be started with the base MySQL image available on Docker Hub. Simply run the following command.

```bash
docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
```

#### 3. Start DeepSparse Server

DeepSparse not only includes high performance runtime on CPUs, but also comes with tooling that simplify the process of adding inference to an application. Once example of this is the Server functionality, which makes it trivial to stand up a model service using DeepSparse.

We have provided a configuration file in `/server/deepsparse-server/server-config-deepsparse.yaml`, which sets up an embedding extraction endpoint running a sparse version of BERT from SparseZoo. You can edit this file to adjust the number of workers you want (this is the number of concurrent inferences that can occur). Generally, its a fine starting point to use `num_cores/2`.

Here's what the config file looks like.

```yaml
num_workers: 4  # number of streams - should be tuned, num_cores / 2 is good place to start

endpoints: 
  - task: embedding_extraction
    model: zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni
    route: /predict
    name: embedding_extraction_pipeline
    kwargs:
      return_numpy: False
      extraction_strategy: reduce_mean
      sequence_length: 512
      engine_type: deepsparse
```

To start DeepSparse, run the following:

```bash
deepsparse.server --config_file server/deepsparse-server/server-config-deepsparse.yaml
```

TO BE REMOVED --- hack to remove bug in Server

- Run `vim deepsparse-env/lib/python3.8/site-packages/deepsparse/server/server.py`
- In `_add_pipeline_endpoint()`, udpate `app.add_api_route` by commenting out `response_model=output_schema`.

ESC-I enters insert mode; ESC Exits insert mode. :wq writes file and quits.

**Potential Improvements**

There is both a throughput-focused step (`load`) where we need to process a large number of embeddings at once with no latency requirements and there is a latency-focused step (`search`) where we need to process one embedding and return to the user as fast as possible. For simplicity, we currently only use one configuration of DeepSparse with `batch_size=1`, which is a latency-oriented setup.

An extension to this project would be configuring DeepSparse to have multiple endpoints or adding another DeepSparse Server instance with a configuration for high throughput.

#### 4. Start The App Server

The App Server is built on `FastAPI` and `uvicorn` and orchestrates DeepSparse, Milvus, and MySQL to create a search engine. 

Run the following to launch.

```bash
python3 server/app-server/src/app.py
```

### Use the Search Engine!

We have provided both a Jupyter notebook and latency testing script to interact with the server. 

#### Jupyter Notebook
The Jupyter notebook is self-documenting and is a good starting point to play around with the application.

You can run with the following command:
`juptyer notebook example-client.ipynb`

#### Latency Testing Script
The latency testing script generates multiple clients to test response time from the server. It provides metrics on both overall query latency as well as metrics on the model serving query latency (the end to end time from the app server querying DeepSparse until a response is returned.) 

You can run with the following command:
```bash
python3 client/latency-test-client.py --url http://localhost:5000/ --dataset_path client/example.csv --num_clients 8
```
- `--url` is the location of the app server
- `--dataset_path` is the location of the dataset path on client side
- `--num_clients` is the number of clients that will be created to send requests concurrently

## Running in an AWS VPC with Independent-Scaling

### Create a VPC

First, we will create a VPC that houses our instances and enables us to communicate between the App Server, Milvus, MySQL, and DeepSparse.

- Navigate to `Create VPC` in the AWS console
- Select `VPC and more`. Name it `semantic-search-demo-vpc`
- Make sure you have `IPv4 CIDR block` set. We use `10.0.0.0/16` in the example.
- Number of AZs to 1, Number of Public Subnets to 1, and Number of Private Subnets to 0.

When we create our services, we will add them to the VPC and only enable communication to the backend model service and databases from within the VPC, isloating the model and database services from the internet.

### Create a Database Instance

Launch an EC2 Instance.
- Navigate to EC2 > Instances > Launch an Instance
- Name the instance `database-server`
- Select Amazon Linux

Edit the `Network Setting`.
- Put the `app-server` into the `semantic-search-demo-vpc` VPC
- Choose the public subnet
- Set `Auto-Assign Public IP` to `Enabled`.
- Add a `Custom TCP` security group rule with port `19530` with `source-type` of `Custom` and Source equal to the CIDR of the VPC (in our case `10.0.0.0/16`). This is how the App Server will Talk to Milvus
- Add a `Custom TCP` security group rule with port `3306` with `source-type` of `Custom` and Source equal to the CIDR of the VPC (in our case `10.0.0.0/16`). This is how the App Server will Talk to MySQL

Launch the instance and then SSH into your newly created instance and start-up the app server.
```
ssh -i path/to/your/private-key.pem ec2-user@your-instance-public-ip
```
Install Docker/Docker Compose and add group membership for the default ec2-user:
```
sudo yum update -y
sudo yum install docker -y
sudo usermod -a -G docker ec2-user
id ec2-user
newgrp docker
pip3 install --user docker-compose
```

Start Docker and Check it is running with the following:
```
sudo service docker start
docker container ls
```

Download Milvus Docker Image and Launch Milvus with `docker-compose`:
```
wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml -O docker-compose.yml
docker-compose up
```

SSH from another terminal into the same instance to setup MySQL.
```
ssh -i path/to/your/private-key.pem ec2-user@your-instance-public-ip
```

Run the following to launch MySQL:
```bash
docker run -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 -d mysql:5.7
```

Your databases are up and running!

### Create the Application Server

Launch an EC2 Instance.
- Navigate to EC2 > Instances > Launch an Instance
- Name the instance `app-server`
- Select Amazon Linux

Edit the `Network Setting` to expose the App Endpoint to the Internet while still giving access to the backend database and model service.
- Put the `app-server` into the `semantic-search-demo-vpc` VPC
- Choose the public subnet
- Set `Auto-Assign Public IP` to `Enabled`.
- Add a `Custom TCP` security group rule with port `5000` with `source-type` of `Anywhere`. This exposes the app to the internet.

Click Launch Instance and SSH into your newly created instance and launch the app server.

From the command line run:
```
ssh -i path/to/your/private-key.pem ec2-user@your-instance-public-ip
```

Clone this repo with Git:
```bash
sudo yum update -y
sudo yum install git -y
sudo git clone https://github.com/rsnm2/deepsparse-milvus.git
```

Install App Requirements in a virutal enviornment.
```bash
python3 -m venv app-env
source app-env/bin/activate
pip3 install -r deepsparse-milvus/text-search-engine/server/app-requirements.txt
```

Run the following to activate.
```bash
python3 deepsparse-milvus/text-search-engine/server/app-server/src/app.py --database host private.ip.of.database.server --model_host private.ip.of.model.server
```

Your App Server is up and Running!

### Create DeepSparse AWS Instance

Launch an EC2 Instance.
- Navigate to EC2 > Instances > Launch an Instance
- Name the instance `database-server`
- Select Amazon Linux and a `c6i.4xlarge` instance type

Edit the `Network Setting` to expose the App Endpoint to the Internet while still giving access to the backend database and model service.
- Put the `app-server` into the `semantic-search-demo-vpc` VPC
- Choose the public subnet
- Set `Auto-Assign Public IP` to `Enabled`.
- Add a `Custom TCP` security group rule with port `5543` with `source-type` of `Custom` and Source equal to the CIDR of the VPC (in our case `10.0.0.0/16`). This is how the App Server will Talk to DeepSparse

Click Launch Instance and SSH into your newly created instance and launch the DeepSparse Server.
```
ssh -i path/to/your/private-key.pem ec2-user@your-instance-public-ip
```

Clone this repo with Git:
```bash
sudo yum update -y
sudo yum install git -y
git clone https://github.com/rsnm2/deepsparse-milvus.git
```

Install App Requirements in a virutal enviornment.
```bash
python3 -m venv deepsparse-env
source deepsparse-env/bin/activate
pip3 install -r deepsparse-milvus/text-search-engine/server/deepsparse-requirements.txt
```

TO BE REMOVED --- hack to remove bug in Server

- Run `vim deepsparse-env/lib/python3.7/site-packages/deepsparse/server/server.py`
- In `_add_pipeline_endpoint()`, udpate `app.add_api_route` by commenting out `response_model=output_schema`.


Run the following to start a model server with DeepSparse as the runtime engine. 
```bash
deepsparse.server --config-file deepsparse-milvus/text-search-engine/server/deepsparse-server/server-config-onnxruntime.yaml```
```

You should see a Uvicorn server running!

We have also provided a config file with ONNX as the runtime engine for performance comparison. 
You can launch a server with ONNX Runtime with the following:
```bash
deepsparse.server --config-file deepsparse-milvus/text-search-engine/server/deepsparse-server/server-config-onnx.yaml
```
**Note: you should have either DeepSparse or ONNXRuntime running but not both***

### Benchmark Performance

From your local machine, run the following, which creates 4 clients that continously make requests to the server.

```bash
python3 client/latency-test-client.py --url http://app-server-public-ip:5000/ --dataset_path client/example.csv --num_clients 4 --iters_per_client 25
```

With DeepSparse running in the Model Server, the latency looks like this, where Model Latency is the time it takes to process
a request by Model Server and Query Latency is the full end to end time on the client side (Network Latency + Model Latency + Database Latency).

```
Model Latency Stats:
{'count': 100,
 'mean': 97.6392858400186,
 'median': 97.46583750006721,
 'std': 0.7766356131548698}

Query Latency Stats:
{'count': 100,
 'mean': 425.1315195999632,
 'median': 425.0526745017851,
 'std': 34.73163016766087}
```

**RS Note: when scaling this out with more clients, the rest of the system becomes the bottleneck for scaling. So, need to investigate a bit more how to show off the performance of DeepSparse**
