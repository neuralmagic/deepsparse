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
Client object for making requests to the DeepSparse inference server
"""

import requests


__all__ = [
    "PipelineClient",
]


class PipelineClient:
    """
    Client object for making requests to the example DeepSparse BERT inference server

    :param address: IP address of the server, default is 0.0.0.0
    :param port: Port the server is hosted on, default is 5543
    """

    def __init__(self, address: str = "0.0.0.0", port: str = "5543"):
        self._url = f"http://{address}:{port}/predict"

    def __call__(self, **kwargs):
        """
        :param kwargs: named inputs to the model server pipeline. e.g. for
            question-answering - `question="...", context="..."`
        :return: json outputs from running the model server pipeline with the
            given input(s)
        """
        # Send data to server for inference
        response = requests.post(self._url, json=kwargs)
        return response.json()
