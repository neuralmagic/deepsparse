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

### INTRO

[GCP's Cloud Run](https://cloud.google.com/run) is a serverless, event-driven environment for making quick deployments for various applications including machine learning in varous programming languages. The most convenient Cloud Run feature is delegating server management to GCP's infrastructure allowing the developer to focus on the deployment with minimum management.

[Getting Started with the DeepSparse Server](https://github.com/neuralmagic/deepsparse)🔌

### Requirements

The listed steps can be easily completed using `Python` and `Bash`. The following
credentials, tools, and libraries are also required:
* The [gcloud CLI](https://cloud.google.com/sdk/gcloud)
* [Docker and the `docker` cli](https://docs.docker.com/get-docker/).

**Before starting, replace the `billing_id` PLACEHOLDER string with your GCP billing ID at the bottom of the SparseRun class found in the `endpoint.py` file. Your billing id, found in the `BILLING` menu in your GCP console should be alphanumeric look something like this:** `XXXXX-XXXXX-XXXXX`

### Quick Start

```bash
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/google-cloud-run
```
Run the following command to build your Cloud Run endpoint.

```bash
python endpoint.py create
```

After the endpoint has been staged (~3 minute), gcloud CLI will provide your API endpoint URL. You can start making requests by passing this URL into the CloudRunClient object. Afterwards, you can run inference by passing in your text input:

```python
from client import CloudRunClient

CR = CloudRunClient("https://1zkckuuw1c.execute-api.us-east-1.amazonaws.com/inference")
answer = CR.client("Drive from California to Texas!")
print(answer)
```
`[{'entity': 'LABEL_0','word': 'drive', ...}, 
{'entity': 'LABEL_0','word': 'from', ...}, 
{'entity': 'LABEL_5','word': 'california', ...}, 
{'entity': 'LABEL_0','word': 'to', ...}, 
{'entity': 'LABEL_5','word': 'texas', ...}, 
{'entity': 'LABEL_0','word': '!', ...}]`

On your first cold start, it will take a ~60 seconds to get your first inference, but afterwards, it should be in milliseconds.

If you want to delete your Cloud Run endpoint, run:

```bash
python endpoint.py destroy
```
