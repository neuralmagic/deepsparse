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

# Deploying DeepSparse with Amazon SageMaker

[Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/index.html)
offers an easy-to-use infrastructure for deploying deep learning models at scale.
This directory provides a guided example for deploying a 
[DeepSparse](https://github.com/neuralmagic/deepsparse) inference server on SageMaker for the question answering NLP task.
Deployments benefit from both sparse-CPU acceleration with
DeepSparse and automatic scaling from SageMaker.

### Requirements
The listed steps can be easily completed using a `python` and `bash`. The following
credentials, tools, and libraries are also required:
* The [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) version 2.X that is [configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html). Double check if the `region` that is configured in your AWS CLI matches the region in the SparseMaker class found in the `endpoint.py` file. Currently, the default region being used is `us-east-1`.
* The [ARN](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) of your AWS role that has access to full SageMaker permissions. In the following steps, we will refer to this as `ROLE_ARN`. It should take the form `"arn:aws:iam::XXX:role/service-role/XXX"`. In addition to role permissions, make sure the AWS user also has SageMaker/ECR permissions.
* [Docker and the `docker` cli](https://docs.docker.com/get-docker/).
* The `boto3` python AWS sdk (`pip install boto3`).

### Quick Start

```bash
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/aws-sagemaker
pip install -r requirements.txt
```

**Before starting, replace the `role_arn` PLACEHOLDER string with your AWS [ARN](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) at the bottom of SparseMaker class on the `endpoint.py` file. Your ARN should look something like this:** `"arn:aws:iam::XXX:role/service-role/XXX"`

Run the following command to build your SageMaker endpoint.

```bash
python endpoint.py create
```

After the endpoint has been staged (~1 minute), you can start making requests by passing your endpoint `region name` and your `endpoint name`. Afterwards you can run inference by passing in your question and context:


```python
from qa_client import Endpoint


qa = Endpoint("us-east-1", "question-answering-example-endpoint")
answer = qa.predict(question="who is batman?", context="Mark is batman.")

print(answer)
```

answer: `b'{"score":0.6484262943267822,"answer":"Mark","start":0,"end":4}'`

If you want to delete your endpoint, please use:

```bash
python endpoint.py destroy
```

Continue reading to learn more about the files in this directory, the build requirements, and a descriptive step-by-step guide for launching a SageMaker endpoint.

## Contents
In addition to the step-by-step instructions below, the directory contains
additional files to aid in the deployment.

### Dockerfile
The included `Dockerfile` builds an image on top of the standard `python:3.8` image
with `deepsparse` installed and creates an executable command `serve` that runs
`deepsparse.server` on port 8080. SageMaker will execute this image by running
`docker run serve` and expects the image to serve inference requests at the
`invocations/` endpoint.

For general customization of the server, changes should not need to be made
to the Dockerfile, but to the `config.yaml` file that the Dockerfile reads from
instead.

### config.yaml
`config.yaml` is used to configure the DeepSparse server running in the Dockerfile.
The config must contain the line `integration: sagemaker` so
endpoints may be provisioned correctly to match SageMaker specifications.

Notice that the `model_path` and `task` are set to run a sparse-quantized
question-answering model from [SparseZoo](https://sparsezoo.neuralmagic.com/).
To use a model directory stored in `s3`, set `model_path` to `/opt/ml/model` in
the config and add `ModelDataUrl=<MODEL-S3-PATH>` to the `CreateModel` arguments.
SageMaker will automatically copy the files from the s3 path into `/opt/ml/model`
which the server can then read from.

### push_image.sh

Bash script for pushing your local Docker image to the AWS ECR repository.

### endpoint.py

Contains the SparseMaker object for automating the build of a SageMaker endpoint from a Docker Image. You have the option to customize the parameters of the class in order to match the prefered state of your deployment.

### qa_client.py

Contains a client object for making requests to the SageMaker inference endpoint for the question answering task.
____
More information on the DeepSparse server and its configuration can be found
[here](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server#readme).

## Deploying to SageMaker
The following steps are required to provision and deploy DeepSparse to SageMaker
for inference:
* Build the DeepSparse-SageMaker `Dockerfile` into a local docker image
* Create an [Amazon ECR](https://aws.amazon.com/ecr/) repository to host the image
* Push the image to the ECR repository
* Create a SageMaker `Model` that reads from the hosted ECR image
* Build a SageMaker `EndpointConfig` that defines how to provision the model deployment
* Launch the SageMaker `Endpoint` defined by the `Model` and `EndpointConfig`

### Requirements
The listed steps can be easily completed using a `python` and `bash`. The following
credentials, tools, and libraries are also required:
* The [`aws` cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) that is [configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)
* The [ARN of an AWS role](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) your user has access to that has full SageMaker and ECR permissions. In the following steps, we will refer to this as `ROLE_ARN`. It should take the form `"arn:aws:iam::XXX:role/service-role/XXX"`
* [Docker and the `docker` cli](https://docs.docker.com/get-docker/)
* The `boto3` python AWS sdk (`pip install boto3`)

### Building the DeepSparse-SageMaker Image Locally
The `Dockerfile` can be build from this directory from a bash shell using the following command.
The image will be tagged locally as `deepsparse-sagemaker-example`.

```bash
docker build -t deepsparse-sagemaker-example .
```

### Creating an ECR Repository
The following code snippet can be used in Python to create an ECR repository.
The `region_name` can be swapped to a preferred region. The repository will be named
`deepsparse-sagemaker`.  If the repository is already created, this step may be skipped.

```python
import boto3

ecr = boto3.client("ecr", region_name='us-east-1')
create_repository_res = ecr.create_repository(repositoryName="deepsparse-sagemaker")
```

### Pushing the Local Image to the ECR Repository
Once the image is built and the ECR repository is created, the image can be pushed using the following
bash commands.

```bash
account=$(aws sts get-caller-identity --query Account | sed -e 's/^"//' -e 's/"$//')
region=$(aws configure get region)
ecr_account=${account}.dkr.ecr.${region}.amazonaws.com

aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_account
fullname=$ecr_account/deepsparse-sagemaker:latest

docker tag deepsparse-sagemaker-example:latest $fullname
docker push $fullname
```

An abbreviated successful output will look like:
```
Login Succeeded
The push refers to repository [XXX.dkr.ecr.us-east-1.amazonaws.com/deepsparse-example]
3c2284f66840: Preparing
08fa02ce37eb: Preparing
a037458de4e0: Preparing
bafdbe68e4ae: Preparing
a13c519c6361: Preparing
6817758dd480: Waiting
6d95196cbe50: Waiting
e9872b0f234f: Waiting
c18b71656bcf: Waiting
2174eedecc00: Waiting
03ea99cd5cd8: Pushed
585a375d16ff: Pushed
5bdcc8e2060c: Pushed
latest: digest: sha256:XXX size: 3884
```

### Creating a SageMaker Model
A SageMaker `Model` can now be created referencing the pushed image.
The example model will be named `question-answering-example`.
As mentioned in the requirements, `ROLE_ARN` should be a string arn of an AWS
role with full access to SageMaker.

```python
sm_boto3 = boto3.client("sagemaker", region_name="us-east-1")

region = boto3.Session().region_name
account_id = boto3.client("sts").get_caller_identity()["Account"]

image_uri = "{}.dkr.ecr.{}.amazonaws.com/deepsparse-sagemaker:latest".format(account_id, region)

create_model_res = sm_boto3.create_model(
    ModelName="question-answering-example",
    Containers=[
        {
            "Image": image_uri,
        },
    ],
    ExecutionRoleArn=ROLE_ARN,
    EnableNetworkIsolation=False,
)
```

More information about options for configuring SageMaker `Model` instances can
be found [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html).


### Building a SageMaker EndpointConfig
The `EndpointConfig` is used to set the instance type to provision, how many, scaling
rules, and other deployment settings.  The following code snippet defines an endpoint
with a single machine using an `ml.c5.large` CPU.

* [Full list of available instances](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html) (See Compute optimized (no GPUs) section)
* [EndpointConfig documentation and options](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html)

```python
model_name = "question-answering-example"  # model defined above
initial_instance_count = 1
instance_type = "ml.c5.2xlarge" # 8 vcpus

variant_name = "QuestionAnsweringDeepSparseDemo"  # ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}

production_variants = [
    {
        "VariantName": variant_name,
        "ModelName": model_name,
        "InitialInstanceCount": initial_instance_count,
        "InstanceType": instance_type,
    }
]

endpoint_config_name = "QuestionAnsweringExampleConfig"  # ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}

endpoint_config = {
    "EndpointConfigName": endpoint_config_name,
    "ProductionVariants": production_variants,
}

endpoint_config_res = sm_boto3.create_endpoint_config(**endpoint_config)
```

### Launching a SageMaker Endpoint
Once the `EndpointConfig` is defined, the endpoint can be easily launched using
the `create_endpoint` command:

```python
endpoint_name = "question-answering-example-endpoint"
endpoint_res = sm_boto3.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
```

After creating the endpoint, its status can be checked by running the following.
Initially, the `EndpointStatus` will be `Creating`. Checking after the image is
successfully launched, it will be `InService`. If there are any errors, it will 
become `Failed`.

```python
from pprint import pprint
pprint(sm_boto3.describe_endpoint(EndpointName=endpoint_name))
```


## Making a Request to the Endpoint
After the endpoint is in service, requests can be made to it through the
`invoke_endpoint` api. Inputs will be passed as a JSON payload.

```python
import json

sm_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

body = json.dumps(
    dict(
        question="Where do I live?",
        context="I am a student and I live in Cambridge",
    )
)

content_type = "application/json"
accept = "text/plain"

res = sm_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=body,
    ContentType=content_type,
    Accept=accept,
)

print(res["Body"].readlines())
```


### Cleanup
The model and endpoint can be deleted with the following commands:
```python
sm_boto3.delete_endpoint(EndpointName=endpoint_name)
sm_boto3.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
sm_boto3.delete_model(ModelName=model_name)
```

## Next Steps
These steps create an invokable SageMaker inference endpoint powered by the DeepSparse
Engine.  The `EndpointConfig` settings may be adjusted to set instance scaling rules based
on deployment needs.

More information on deploying custom models with SageMaker can be found
[here](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).

Open an [issue](https://github.com/neuralmagic/deepsparse/issues)
or reach out to the [DeepSparse community](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
with any issues, questions, or ideas.
