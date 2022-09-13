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

import requests


class LambdaClient:
    """
    Client object for making requests to the Lambda HTTP endpoint
    :param url: API endpoint URL
    """

    def __init__(self, url: str):

        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def qa_client(self, question: str, context: str) -> bytes:

        """
        :param question: question input to the model pipeline.
        :param context: context input to the model pipeline.
        :return: json output from Lambda
        """

        obj = {"question": question, "context": context}

        response = requests.post(self.url, headers=self.headers, json=obj)

        return json.loads(response.content)
