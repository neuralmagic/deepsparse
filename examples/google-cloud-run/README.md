<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Deploying the DeepSparse Server with GCP's Cloud Run

[GCP's Cloud Run](https://cloud.google.com/run) is a serverless, event-driven environment for making quick deployments for various applications including machine learning in various programming languages. The most convenient Cloud Run feature is delegating server management to GCP's infrastructure allowing the developer to focus on the deployment with minimal management.

[Getting Started with the DeepSparse Server](https://github.com/neuralmagic/deepsparse)🔌

## Requirements

The listed steps can be easily completed using `Python` and `Bash`. The following
credentials, tools, and libraries are also required:
* The [gcloud CLI](https://cloud.google.com/sdk/gcloud)
* [Docker and the `docker` cli](https://docs.docker.com/get-docker/).

**Before starting, replace the `billing_id` PLACEHOLDER string with your GCP billing ID at the bottom of the SparseRun class in the `endpoint.py` file. Your billing id, can be found in the `BILLING` menu of your GCP console. It should be alphanumeric and look something like this:** `XXXXX-XXXXX-XXXXX`



## Quick Start

```bash
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/google-cloud-run
```

## Model Configuration

They current server configuration is running `token classification`. To alter the model, task or other parameters (i.e., num. of cores, workers, batch size etc.), please edit the `config.yaml` file.

## Create Endpoint
Run the following command to build your Cloud Run endpoint.

```bash
python endpoint.py create
```
## Call Endpoint

After the endpoint has been staged (~3 minute), gcloud CLI will output your API Service URL. You can start making requests by passing this URL AND its route into the CloudRunClient object. The route can be configured in the `config.yaml`. 

For example, if your Service URL is `https://deepsparse-cloudrun-qsi36y4uoa-ue.a.run.app` and your route is `/inference`, the complete URL passed into the client would be: `https://deepsparse-cloudrun-qsi36y4uoa-ue.a.run.app/inference`


Afterwards, you can call your endpoint by passing in your text input:

```python
from client import CloudRunClient

CR = CloudRunClient("https://sparserun-xxxxxxxxxx-xx.a.run.app/{ROUTE}")
answer = CR.client("Drive from California to Texas!")
print(answer)
```
`[{'entity': 'LABEL_0','word': 'drive', ...}, 
{'entity': 'LABEL_0','word': 'from', ...}, 
{'entity': 'LABEL_5','word': 'california', ...}, 
{'entity': 'LABEL_0','word': 'to', ...}, 
{'entity': 'LABEL_5','word': 'texas', ...}, 
{'entity': 'LABEL_0','word': '!', ...}]`

Additionally, you can also call the endpoint via a cURL command:

```bash
curl -X 'POST' \
  'https://sparserun-xxxxxxxxxx-xx.a.run.app/{ROUTE}' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    "Drive from California to Texas!"
  ],
  "is_split_into_words": false
}'
```

FYI, on your first cold start, it will take a ~60 seconds to get your first inference, but afterwards, it should be in milliseconds.

## Delete Endpoint

If you want to delete your Cloud Run endpoint, run:

```bash
python endpoint.py destroy
```
