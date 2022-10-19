# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script for auto-generating a SageMaker endpoint from
a local Docker image

##########
Command help:
usage: python endpoint.py [-h] [action]

Args:
  action [create, destroy]  choose between creating or destroying an endpoint

##########
Example command for creating an endpoint:

python endpoint.py create

Example command for destroying an endpoint:

python endpoint.py destroy
"""
import subprocess

import click

import boto3
from rich.pretty import pprint


class SparseMaker:

    """
    Object for auto generating a docker image with the deepsparse server,
    pushing the image to AWS and generating a SageMaker inference endpoint

    :param instance_count: Number of instances to launch initially
    :param region_name: AWS region
    :param instance_type: Type of instance to spin up on sagemaker
    :param image_name: Image name given to Docker when building dockerfile
    :param repo_name: ECR repo name on AWS
    :param image_push_script: path to script for pushing Docker Image to ECR
    :param variant_name: The name of the production variant
    :param model_name: The name of the model that you want to host.
                       This is the name that you specified when creating the model
    :param config_name: The name of the endpoint configuration
                        associated with the endpoint
    :param endpoint_name: name given to endpoint on SageMaker
    :param role_arn: ARN id that gives SageMaker and ECR permissions

    """

    def __init__(
        self,
        instance_count: int,
        region_name: str,
        instance_type: str,
        image_name: str,
        repo_name: str,
        image_push_script: str,
        variant_name: str,
        model_name: str,
        config_name: str,
        endpoint_name: str,
        role_arn: str,
    ):

        # Docker Image and Repository
        self.image_name = image_name
        self.build_cmd = [f"docker build -t {self.image_name} ."]
        self.repo_name = repo_name
        self.push_script = image_push_script

        # Sagemaker Client
        self.region_name = region_name
        self.sm_boto3 = boto3.client("sagemaker", region_name=self.region_name)

        # Model
        self.acc = boto3.client("sts").get_caller_identity()["Account"]
        self.repo_path = f"amazonaws.com/{self.repo_name}:latest"
        self.image_uri = f"{self.acc}.dkr.ecr.{self.region_name}.{self.repo_path}"
        self.image = [{"Image": self.image_uri}]
        self.ROLE_ARN = role_arn

        # Endpoint Configuration
        self.variant_name = variant_name
        self.model_name = model_name
        self.initial_instance_count = instance_count
        self.instance_type = instance_type
        self.production_variants = [
            {
                "VariantName": self.variant_name,
                "ModelName": self.model_name,
                "InitialInstanceCount": self.initial_instance_count,
                "InstanceType": self.instance_type,
            }
        ]
        self.endpoint_config_name = config_name
        self.endpoint_config = {
            "EndpointConfigName": self.endpoint_config_name,
            "ProductionVariants": self.production_variants,
        }
        self.endpoint_name = endpoint_name

    def create_image(self):

        subprocess.run(self.build_cmd, shell=True)

    def create_ecr_repo(self):

        try:
            ecr = boto3.client("ecr", region_name=self.region_name)
            ecr.create_repository(repositoryName=self.repo_name)

        except ecr.exceptions.RepositoryAlreadyExistsException:
            pass

        repo_check = ecr.describe_repositories(repositoryNames=[self.repo_name])
        pprint(repo_check["repositories"])

    def push_image(self):

        subprocess.call(["sh", self.push_script])

    def create_model(self):

        self.sm_boto3.create_model(
            ModelName=self.model_name,
            Containers=self.image,
            ExecutionRoleArn=self.ROLE_ARN,
            EnableNetworkIsolation=False,
        )

    def create_endpoint_config(self):

        self.sm_boto3.create_endpoint_config(**self.endpoint_config)

    def create_endpoint(self):

        self.sm_boto3.create_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=self.endpoint_config_name,
        )

        pprint(self.sm_boto3.describe_endpoint(EndpointName=self.endpoint_name))

    def destroy_endpoint(self):

        self.sm_boto3.delete_endpoint(EndpointName=self.endpoint_name)
        self.sm_boto3.delete_endpoint_config(
            EndpointConfigName=self.endpoint_config_name
        )
        self.sm_boto3.delete_model(ModelName=self.model_name)

        print("endpoint and SageMaker model deleted")


def construct_sparsemaker():
    return SparseMaker(
        instance_count=1,
        region_name="us-east-1",
        instance_type="ml.c5.2xlarge",
        image_name="deepsparse-sagemaker-example",
        repo_name="deepsparse-sagemaker",
        image_push_script="./push_image.sh",
        variant_name="QuestionAnsweringDeepSparseDemo",
        model_name="question-answering-example",
        config_name="QuestionAnsweringExampleConfig",
        endpoint_name="question-answering-example-endpoint",
        role_arn="<PLACEHOLDER>",
    )


@click.group(chain=True)
def main():
    pass


@main.command("create")
def create():
    SM = construct_sparsemaker()
    SM.create_image()
    SM.create_ecr_repo()
    SM.push_image()
    SM.create_model()
    SM.create_endpoint_config()
    SM.create_endpoint()


@main.command("destroy")
def destroy():
    SM = construct_sparsemaker()
    SM.destroy_endpoint()


if __name__ == "__main__":
    main()
