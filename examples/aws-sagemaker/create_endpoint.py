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

import subprocess

import boto3
from rich.pretty import pprint


class SparseMaker:
    def __init__(
        self,
        instance_count,
        region_name,
        instance_type,
        image_name,
        repository_name,
        image_push_script,
        variant_name,
        model_name,
        config_name,
        endpoint_name,
        role_arn,
    ):

        # Docker Image and Repository
        self.image_name = image_name
        self.build_cmd = [f"docker build -t {self.image_name} ."]
        self.repo_name = repository_name
        self.push_script = image_push_script

        # Sagemaker Client
        self.region_name = region_name
        self.sm_boto3 = boto3.client("sagemaker", region_name=self.region_name)

        # Model
        self.region = boto3.Session().region_name
        self.acc = boto3.client("sts").get_caller_identity()["Account"]
        self.image_uri = f"{self.acc}.dkr.ecr.{self.region}.amazonaws.com/deepsparse-sagemaker:latest"
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

    def nuke_endpoint(self):

        self.sm_boto3.delete_endpoint(EndpointName=self.endpoint_name)
        self.sm_boto3.delete_endpoint_config(
            EndpointConfigName=self.endpoint_config_name
        )
        self.sm_boto3.delete_model(ModelName=self.model_name)

        print("endpoint nuked!")


if __name__ == "__main__":

    Sage = SparseMaker(
        instance_count=1,
        region_name="us-east-1",
        instance_type="ml.c5.large",
        image_name="deepsparse-sagemaker-example",
        repository_name="deepsparse-sagemaker",
        image_push_script="./push_image.sh",
        variant_name="QuestionAnsweringDeepSparseDemo",
        model_name="question-answering-example",
        config_name="QuestionAnsweringExampleConfig",
        endpoint_name="question-answering-example-endpoint",
        role_arn="<placeholder>",
    )

    Sage.create_image()
    Sage.create_ecr_repo()
    Sage.push_image()
    Sage.create_model()
    Sage.create_endpoint_config()
    Sage.create_endpoint()
    Sage.nuke_endpoint()
