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

import json

import boto3


class Endpoint:
    def __init__(self, region_name, endpoint_name):

        self.region_name = region_name
        self.endpoint_name = endpoint_name
        self.content_type = "application/json"
        self.accept = "text/plain"
        self.client = boto3.client("sagemaker-runtime", region_name=self.region_name)

    def predict(self, question, context):

        body = json.dumps(
            dict(
                question=question,
                context=context,
            )
        )
        res = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=body,
            ContentType=self.content_type,
            Accept=self.accept,
        )

        print(res["Body"].readlines())
