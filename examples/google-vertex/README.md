# nm-gcp-vertex
Example using DeepSparse in a GCP Vertex model serving endpoint for Sentiment Analysis

GCP has [great documentation](https://console.cloud.google.com/vertex-ai) on using a custom container to create a Model for usage in inference Endpoints. Let's see how we can use this functionality to run DeepSparse in a Vertex Endpoint!

## Overview

The following steps are required to provision and deploy DeepSparse to GCP Vertex for inference:
- Create a new Project in the GCP Console
- Create an Artifact Repository to host a Docker image with DeepSparse
- Build a Docker Image and push to the Artifact Repository
- Create a Vertex `Model` with the hosted Docker image
- Build a Vertex `Endpoint` for model serving
- Deploy the `Model` to the `Endpoint`

We will use the `gcloud` CLI to for each step.

## 1. Create a New Project In the GCP Console

From the GCP Console, create a new project. We named our project `gcp-vertex-deepsparse-example`.

Set the `PROJECT_ID` enviornment variable on your machine to match the ID of the project. In our case:
```
export PROJECT_ID=gcp-vertex-deepsparse-example
```

Once the project is created, switch to the project on your machine with the `gcloud` CLI command:
```
gcloud config set project $PROJECT_ID
```

Choose the region you want to run in and set it as an environment variable. For example:
```
export REGION=us-east1
```

## 2. Create Artifact Repository

- Authenticate to gcloud
```
gcloud auth configure-docker $REGION-docker.pkg.dev
```

- Enable Artifact Repository API
```
gcloud services enable artifactregistry.googleapis.com
```
Wait a few minutes for this to propogate through GCP's systems.

- Create Artifact Repository with Docker format
```
gcloud artifacts repositories create deepsparse-server-images --repository-format=docker --location=$REGION
```

- Build Docker Image Locally
We provided a Dockerfile which downloads DeepSparse and launches the server based on the `server-config.yaml` file provided. If you want to use a different model or Server configuration, update the `server-config.yaml` as needed.
```
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/deepsparse-server-images/sentiment-analysis .
```

Push Docker Image to Artifact Repository
```
docker push $REGION-docker.pkg.dev/$PROJECT_ID/deepsparse-server-images/sentiment-analysis
```

## Setup Model and Endpoint

#### Enable Vertex AI

Unfortunately there is no gcloud CLI command to enable the Vertex API. As such, bavigate to the [Vertex AI Dashboard](https://console.cloud.google.com/vertex-ai) and click "Enable Recommended APIs."

#### Create a Model: 

The [GCP Docs](https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container#examples) specify a few arguments that must be passed, including the region, the location of the container in the Artifact Respository, the port used by the Server in the container, the health route, and the prediction route. In the case of DeepSparse Server, we configured it to use port 5543 and the routes `/health` and `/predict` (these are the defaults) - so we will pass these values when creating the Model.

```bash
gcloud ai models upload --region=$REGION --display-name=sparse-sentiment-analysis --container-image-uri=$REGION-docker.pkg.dev/$PROJECT_ID/deepsparse-server-images/sentiment-analysis --container-ports=5543 --container-health-route=/health --container-predict-route=/predict
```

This command may fail with an error about IAM policies:
```bash
ERROR: (gcloud.ai.models.upload) FAILED_PRECONDITION: Vertex AI Service Agent service-XXX@gcp-sa-aiplatform.iam.gserviceaccount.com does not have permission to access Artifact Registry repository projects/gcp-vertex-deepsparse-example/locations/us-east1/repositories/deepsparse-server-images.
```

Set the Service Agent as an enviornment variable:
```bash
export SERVICE_AGENT=service-XXX@gcp-sa-aiplatform.iam.gserviceaccount.com
```

Grant access to the repository as a reader:
```bash
gcloud artifacts repositories add-iam-policy-binding deepsparse-server-images --location $REGION --member=serviceAccount:$SERVICE_ACCOUNT --role=roles/artifactregistry.reader
```

Re-running the `upload` command should now work.

#### Create An Endpoint

Create an Endpoint
```bash
gcloud ai endpoints create --region=$REGION --display-name=deepsparse-endpoint
```

This will take a few minutes to complete.

#### Deploy Model to the Endpoint

Get Model and Endpoint IDs
```bash
gcloud ai models list --region=$REGION
gcloud ai endpoints list --region=$REGION
```

Save the numbers that appear in the `ENDPOINT_ID` and `MODEL_ID` columns as enviornment variables, such as:

```bash
export ENDPOINT_ID=5488427794322948096
export MODEL_ID=4905880145648156672
```

```bash
gcloud ai endpoints deploy-model $ENDPOINT_ID --region=us-east1 --model=$MODEL_ID --display-name=sparse-model --machine-type=n1-highcpu-8 --min-replica-count=1
```

Save the Id of the deployed model as an enviornment variable:
```
export DEPLOYED_MODEL_ID=7899507260454338560
```

This will take a few minutes to complete.

Note that our endpoint does not use any accelerators and is running on a standard CPU-only instance.

We are ready to start performing inferences!

## Send Requests To The Server

The [GCP Raw Prediction API](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions)  enables you to send aribtray HTTP payload.

Create a JSON file to send to the server. As per the [documentation](https://docs.neuralmagic.com/use-cases/natural-language-processing/deploying), the Sentiment Analysis Pipeline expects an array of sequences.

```json
{"sequences": ["The man dislikes going to the store", "The man loves going to the store"]}
```

Send A Request to the Endpoint:
```
gcloud ai endpoints raw-predict $ENDPOINT_ID --region=$REGION --request=@request.json

# {"labels":["LABEL_0","LABEL_1"],"scores":[0.9987145662307739,0.9933835864067078]}
```

We successfully made a prediction!

## Cleaning Up

Undeploy your Model from the Endpoint.
```
gcloud ai endpoints undeploy-model $ENDPOINT_ID --region=$REGION --deployed-model-id $DEPLOYED_MODEL_ID
```

Delete your Endpoint.
```
gcloud ai endpoints delete $ENDPOINT_ID
```

Delete your Model.
```
gcloud ai models delete $MODEL_ID
```

## Next Steps

Refer to the [GCP documentation](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions) for more inforrmation on deploying custom models with Vertex.

Refer to the [GCP documentation](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute) for more information on compute resources and autoscaling your endpoints.