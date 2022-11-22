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


class CloudRunClient:
    """
    Client object for making requests to the CLoud Run HTTP endpoint
    :param url: API endpoint URL
    """

    def __init__(self, url: str):

        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def client(self, payload):
        """
        Client for NLP tasks.
        """

        response = requests.post(self.url, headers=self.headers, json=payload)

        return json.loads(response.content)
