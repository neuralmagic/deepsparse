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
Example script for deploying a serverless architecture for batch and realtime inference.

##########
Command help:
usage: python endpoint.py [-h] [action]

Args:
  action [create-batch, create-realtime, destroy, destroy-all]  choose between creating
  a batch inference, realtime inference endpoints, destroying an endpoint, and destroying
  endpoint and S3 buckets.

##########
Example command for creating a batch inference architecture:

python endpoint.py create-batch

Example command for destroying either a batch or realtime inference endpoint:

python endpoint.py destroy

Example command for destroying either a batch or realtime inference endpoint
and S3 buckets (if they exist):

python endpoint.py destroy-all
"""


class SparseLambda:

    """
    Object for generating a batch or realtime inference deployment on AWS
    serverless via AWS SAM and cloudformation orchestration.

    :param region_name: AWS region
    :param ecr_repo_name: ECR repo name on AWS
    :param stack_name: cloudformation name on AWS

    """

    def __init__(self, region_name: str, ecr_repo_name: str, stack_name: str):

        self.region_name = region_name
        self.ecr_repo_name = ecr_repo_name
        self.stack_name = stack_name

        self.batch_script = "./scripts/batch-startup.sh"
        self.realtime_script = "./scripts/realtime-startup.sh"
        self.ecr = boto3.client("ecr", region_name=self.region_name)
        self.cloudformation = boto3.client("cloudformation")
        self.s3 = boto3.resource('s3')

    def create_ecr_repo(self):

        try:
            self.ecr.create_repository(repositoryName=self.ecr_repo_name)

        except self.ecr.exceptions.RepositoryAlreadyExistsException:
            pass

        repo_check = self.ecr.describe_repositories(
            repositoryNames=[self.ecr_repo_name]
        )
        pp.pprint(repo_check["repositories"])

    def create_endpoint(self, batch: bool = True):

        subprocess.call(
            [
                "sh",
                self.batch_script if batch else self.realtime_script,
                self.region_name,
                self.stack_name,
                self.ecr_repo_name,
            ]
        )

    def destroy_endpoint(self):

        # self.cloudformation.delete_stack(StackName=self.stack_name)
        # self.ecr.delete_repository(repositoryName=self.ecr_repo_name, force=True)
        # print(
        #     f"Your '{self.stack_name}' and ECR repo '{self.ecr_repo_name}' were killed."
        # )
        
        self.cloudformation.delete_stack(StackName=self.stack_name)
        print(f"CloudFormation stack '{self.stack_name}' was deleted.")
        
        self.ecr.delete_repository(repositoryName=self.ecr_repo_name, force=True)
        print(f"ECR repository '{self.ecr_repo_name}' was deleted.")

    def destroy_buckets(self):

        for bucket_name in ["batch-input-deepsparse", "batch-output-deepsparse"]:
            bucket = self.s3.Bucket(bucket_name)
            if bucket in self.s3.buckets.all():
                bucket.objects.all().delete()
                bucket.delete()
                
        print(f"S3 buckets and their content were deleted.")


def construct_sparselambda():
    return SparseLambda(
        region_name="us-east-1",
        ecr_repo_name="serverless-deepsparse",
        stack_name="serverless-stack",
    )


@click.group(chain=True)
def main():
    pass


@main.command("create-realtime")
def create_realtime():
    SL = construct_sparselambda()
    SL.create_ecr_repo()
    SL.create_endpoint(batch=False)


@main.command("create-batch")
def create_batch():
    SL = construct_sparselambda()
    SL.create_ecr_repo()
    SL.create_endpoint()


@main.command("destroy")
def destroy():
    SL = construct_sparselambda()
    SL.destroy_endpoint()
  
    
@main.command("destroy-all")
def destroy_all():
    SL = construct_sparselambda()
    SL.destroy_endpoint()
    SL.destroy_buckets()


if __name__ == "__main__":
    main()
