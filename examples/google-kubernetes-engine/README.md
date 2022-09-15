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

# Deploying DeepSparse on Google Kubernetes Engine (GKE)

See [GKE documentation](https://cloud.google.com/kubernetes-engine) for more info on GKE.

## Deploying to GKE

Follow the steps at https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
to deploy the `Dockerfile` in this directory to GKE.

## Sending a request to GKE

After following the tutorial, your web server will be exposed just like a normal web server. You can interact with it just like you would if you ran the docker image locally (e.g. visit the swagger documentation UI with a browser at `/docs` or send requests using curl).  See more information on [interacting with the DeepSparse server here](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server).

## Testing `Dockerfile` Locally

1. `docker build -t qa .`
2. `docker run --rm -p 8080:8080 qa`
3. `curl http://localhost:8080`