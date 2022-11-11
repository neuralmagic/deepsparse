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

import pprint as pp
import subprocess

import click

import boto3


"""
Example script for auto-generating a Lambda HTTP endpoint in AWS Cloud

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


class SparseLambda:

    """
    Object for generating a docker image running a DeepSparse pipeline,
    pushing the image to ECR and generating a Lambda API Gateway endpoint

    :param region_name: AWS region
    :param ecr_repo_name: ECR repo name on AWS
    :param stack_name: cloudformation name on AWS

    """

    def __init__(self, region_name: str, ecr_repo_name: str, stack_name: str):

        self.region_name = region_name
        self.ecr_repo_name = ecr_repo_name
        self.stack_name = stack_name

        self.create_endpoint = "./create_endpoint.sh"
        self.ecr = boto3.client("ecr", region_name=self.region_name)
        self._lambda = boto3.client("lambda", region_name=self.region_name)
        self.cloudformation = boto3.client("cloudformation")

    def create_ecr_repo(self):

        try:
            self.ecr.create_repository(repositoryName=self.ecr_repo_name)

        except self.ecr.exceptions.RepositoryAlreadyExistsException:
            pass

        repo_check = self.ecr.describe_repositories(
            repositoryNames=[self.ecr_repo_name]
        )
        pp.pprint(repo_check["repositories"])

    def create_api_endpoint(self):

        """
        runs bash script for:
        1. building local image
        2. pushing image to ECR
        3. building Lambda API endpoint
        """

        subprocess.call(
            [
                "sh",
                self.create_endpoint,
                self.region_name,
                self.stack_name,
                self.ecr_repo_name,
            ]
        )

    def list_functions(self):

        response = self._lambda.list_functions()

        for function in response["Functions"]:
            print("*** These are your Lambda functions: ***\n")
            print("Function name: " + function["FunctionName"], "\n")

    def destroy_endpoint(self):

        self.cloudformation.delete_stack(StackName=self.stack_name)
        self.ecr.delete_repository(repositoryName=self.ecr_repo_name, force=True)
        print(
            f"Your '{self.stack_name}' and ECR repo '{self.ecr_repo_name}' were killed."
        )


def construct_sparselambda():
    return SparseLambda(
        region_name="us-east-1",
        ecr_repo_name="lambda-deepsparse",
        stack_name="lambda-stack",
    )


@click.group(chain=True)
def main():
    pass


@main.command("create")
def create():
    SL = construct_sparselambda()
    SL.create_ecr_repo()
    SL.create_api_endpoint()
    SL.list_functions()


@main.command("destroy")
def destroy():
    SL = construct_sparselambda()
    SL.destroy_endpoint()


if __name__ == "__main__":
    main()
