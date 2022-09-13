# 🐑 Deploy a DeepSparse Pipeline in AWS Lambda

AWS Lambda is an event-driven, serverless computing infrastructure for deploying applications at minimal cost. This directory provides a guided example for deploying a DeepSparse pipeline on AWS Lambda for the question answering NLP task.

The scope of this application is able to automate:
1. The construction of a local docker image.
2. Create an ECR repo in AWS.
3. Push the image to ECR.
4. Create the appropriate IAM permissions for handling Lambda.
4. Create a Lambda function alongside an API Gateway in a cloudformation stack. 

### Requirements
The following credentials, tools, and libraries are also required:
* The [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) version 2.X that is [configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html). Double check if the `region` that is configured in your AWS CLI matches the region passed in the SparseLambda class found in the `endpoint.py` file. Currently, the default region being used is `us-east-1`.
* The AWS Serverless Application Model [(AWS SAM)](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html), an open-source CLI framework used for building serverless applications on AWS.
* [Docker and the `docker` cli](https://docs.docker.com/get-docker/).
* The `boto3` python AWS SDK: `pip install boto3`.


### Quick Start

```bash 
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/aws-lambda
pip install -r requirements.txt
```

Run the following command to build your Lambda endpoint.

```bash
python endpoint.py create
```

After the endpoint has been staged (~1 minute), AWS SAM will provide your API Gateway endpoint URL in CLI. You can start making requests by passing this URL into the LambdaClient object. Afterwards, you can run inference by passing in your question and context:

```python
from client import LambdaClient

LC = LambdaClient("https://1zkckuuw1c.execute-api.us-east-1.amazonaws.com/inference")
answer = LC.qa_client(question="who is batman?", context="Mark is batman.")

print(answer)
```

answer: `{'Question': 'who is batman?', 'Answer': 'Mark'}`

On your first cold start, it will take a ~30 seconds to get your first inference, but afterwards, it should be in milliseconds.


If you want to delete your Lambda endpoint, run:

```bash
python endpoint.py destroy
```