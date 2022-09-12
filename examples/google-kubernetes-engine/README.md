# Deploying DeepSparse on Google Kubernetes Engine (GKE)

See [GKE documentation](https://cloud.google.com/kubernetes-engine) for more info on GKE.

## Deploying to GKE

Follow the steps at https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
to deploy the `Dockerfile` in this directory to GKE.

## Testing `Dockerfile` Locally

1. `docker build -t qa .`
2. `docker run --rm -p 8080:8080 qa`
3. `curl http://localhost:8080`