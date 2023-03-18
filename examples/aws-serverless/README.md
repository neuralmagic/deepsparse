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

# DeepSparse Inference Using AWS Serverless

![image](./img/aws_serverless_arch.png)

This repo allows users to build a serverless computing infrastructure for deploying inference at scale. This guided example can be used to deploy a DeepSparse pipeline on AWS Lambda for realtime inference or on AWS Fargate for batch inference. This is demonstrated using a sentiment analysis use case.

The scope of this application encompasses:
1. The construction of local Docker images.
2. The creation of an ECR repo in AWS.
3. Pushing the local images to ECR.
4. Deploying a:
   - **Realtime Inference Infrastructure**: the creation of a Lambda function via API Gateway in a Cloudformation stack.

   or

   - **Batch Inference Infrastructure**: the creation of a serverless instance on AWS Fargate via AWS Batch in a Cloudformation stack.

## Requirements
The following credentials, tools, and libraries are also required:
* The [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) version 2.X that is [configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html). Double check if the `region` that is configured in your AWS CLI matches the region passed in the SparseLambda class found in the `endpoint.py` file. Currently, the default region being used is `us-east-1`.
* Full permissions to select AWS resources: ECR, API Gateway, Cloudformation, and Lambda.
   - The IAM permissions for batch inference are auto-generated at startup.
* The AWS Serverless Application Model [(AWS SAM)](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html), an open-source CLI framework used for building serverless applications on AWS.
* [Docker and the `docker` cli](https://docs.docker.com/get-docker/).
* The `boto3` python AWS SDK and the `click` library.

## Model & Pipeline Configuration

To use a different sparse model for batch inference, please edit the model zoo stub in the Dockerfile here: `/batch/app_inf/Dockerfile`. To edit the model for realtime inference, edit here `/realtime/app/Dockerfile`. 

To change pipeline configuration (i.e., change task, engine etc.), edit the pipeline object in either `app.py` files. Both files can be found in the `/realtime/app/` and `/batch/app/` directories.

## Quick Start

```bash 
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/aws-serverless
pip install -r requirements.txt
```

After installation, you can choose to build either a batch or a realtime serverless infrastructure. Both options are detailed below.

## Create Batch Infra

Run the following command to build a batch inference infrastructure:

```bash
python endpoint.py create-batch
```

### Batch Job Flow

After build, upload a CSV file to the `batch-input-deepsparse` S3 bucket (which was auto-generated) via the AWS console or from the following CLI command to start the batch job:

```bash
aws s3 cp <path/to/csv/file> s3://batch-input-deepsparse/ --recursive
```
Afterwards, a Lambda function will trigger a batch job to spin up a Fargate instance running DeepSparse. The CSV file will be read and inputs will be passed into DeepSparse for prediction. Aftewards, the output will be automatically written to a CSV file called `outputs.csv` and pushed to the `batch-output-deepsparse` S3 bucket.

An example `sentiment-inputs.csv` file in the `sample` directory is available to familiarize yourself with the file structure the batch architecture is expecting to receive to perform sentiment analysis.

## Create Realtime Infra

Run the following command to build a realtime infrastructure.

```bash
python endpoint.py create-realtime
```

### Call Realtime Endpoint

After the endpoint has been staged (~3 minute), AWS SAM will provide your API Gateway endpoint URL in CLI. You can start making requests by passing this URL into the LambdaClient object. Afterwards, you can run inference by passing in your text input:

```python
from client import LambdaClient

LC = LambdaClient("https://#########.execute-api.us-east-1.amazonaws.com/inference")
answer = LC.client({"sequences": "i like pizza"})

print(answer)
```

answer: `{'labels': ['positive'], 'scores': [0.9990884065628052]}`

On your first cold start, it will take a ~60 seconds to invoque your first inference, but afterwards, it should be in milliseconds.

## Delete Endpoint

If you want to delete your batch/realtime infrastructure, run:

```bash
python endpoint.py destroy
```