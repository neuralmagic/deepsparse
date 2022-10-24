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


RAND_ID = randint(10000, 99999)


class SparseRun:

    """
    Object for auto generating a docker image with the deepsparse server,
    pushing the image to GCP and generating a Cloud Run inference endpoint

    :param billind_id: Your GCP's account Billind ID
    :param service_name: Name of the cloud run app
    :param image_name: Name of Docker image to be created
    :param region_name: Name of cloud region to use
    """

    def __init__(
        self, billing_id: str, service_name: str, image_name: str, region_name: str
    ):

        self.billing_id = billing_id
        self.service = service_name
        self.image = image_name
        self.region = region_name

        self.project_id = "deepsparse" + str(RAND_ID)
        self.endpoint_script = "./create_endpoint.sh"

        self.del_api = [
            f"gcloud run services delete {self.service} --region {self.region} --quiet"
        ]

    def create_endpoint(self):

        subprocess.call(
            [
                "sh",
                self.endpoint_script,
                self.billing_id,
                self.project_id,
                self.image,
                self.region,
                self.service,
            ]
        )

    def destroy_endpoint(self):

        subprocess.run(self.del_api, shell=True)
        print("Cloud Run endpoint deleted")


def construct_sparserun():
    return SparseRun(
        billing_id="<PLACEHOLDER>",
        service_name="deepsparse-cloudrun",
        image_name="sparserun",
        region_name="us-east1",
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
