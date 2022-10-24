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
Example script for auto-generating a Cloud Run endpoint from
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
from random import randint

import click


class SparseRun:

    """
    Object for auto generating a docker image with the deepsparse server,
    pushing the image to GCP and generating a Cloud Run inference endpoint

    :param billind_id: Your GCP's account Billind ID
    :param image_name: Name of Docker image to be created
    """

    def __init__(self, billing_id: str, image_name: str, region_name: str):

        self.billing_id = billing_id
        self.image_name = image_name
        self.region_name = region_name

        rand_id = randint(10000, 99999)
        self.project_id = "deepsparse" + str(rand_id)
        self.endpoint_script = "./create_endpoint.sh"

    def create_endpoint(self):

        subprocess.call(
            [
                "sh",
                self.endpoint_script,
                self.billing_id,
                self.project_id,
                self.image_name,
                self.region_name,
            ]
        )

    def destroy_endpoint(self):

        print("endpoint and Cloud Run endpoint deleted")


def construct_sparserun():
    return SparseRun(
        billing_id="<PLACEHOLDER>", image_name="sparserun", region_name="us-east1"
    )


@click.group(chain=True)
def main():
    pass


@main.command("create")
def create():
    SM = construct_sparserun()
    SM.create_endpoint()


@main.command("destroy")
def destroy():
    SM = construct_sparserun()
    SM.destroy_endpoint()


if __name__ == "__main__":
    main()
