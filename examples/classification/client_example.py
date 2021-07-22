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
An example client to help make requests to an image classification server
"""
import time
from typing import List, Union

import numpy
import requests

from deepsparse.utils import arrays_to_bytes, bytes_to_arrays
from helper import _BatchLoader, load_data


class Client:
    """
    Client object for making requests to the example DeepSparse inference server

    :param address: IP address of the server, default is 0.0.0.0
    :param port: Port the server is hosted on, default is 5543
    """

    def __init__(self, address: str = "0.0.0.0", port: str = 5543):
        self._url = f"http://{address}:{port}/predict"

    def classify(
        self,
        images: Union[str, numpy.ndarray, List[str], List[numpy.ndarray]],
    ) -> List[numpy.ndarray]:
        """
        :param images: list of numpy arrays of images to
            run the detection model on. Number of images should equal model batch size
        :return: list of post-processed object detection results for each image
            including class label, likelihood, and bounding box coordinates
        """

        if not isinstance(images, List):
            images = [images]

        print(f"Sending batch of {len(images)} images for detection to {self._url}")

        start = time.time()
        # Encode inputs
        data = arrays_to_bytes(images)
        preprocessing_end = time.time()
        preprocessing_time = preprocessing_end - start

        # Send data to server for inference
        response = requests.post(self._url, data=data)
        execution_end_time = time.time()
        engine_execution_time = execution_end_time - preprocessing_end

        # Decode outputs
        outputs = bytes_to_arrays(response.content)
        postprocessing_time = time.time() - execution_end_time
        total_time = time.time() - start

        print(
            f"Preprocessing time {preprocessing_time * 1000:.4f}ms",
            f"Engine run time {engine_execution_time * 1000:.4f}ms",
            f"Postprocessing time {postprocessing_time * 1000:.4f}ms",
            f"Round-trip time  {total_time * 1000.0:.4f}ms",
            sep="\n",
        )

        return outputs


def sanity_check():
    """
    This function shows example usage of Client class,
    run this function after running server.py with the resnet stub

    $python3 server.py MODEL_STUB
    """
    client = Client()
    model_data_stub = (
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
        "imagenet/pruned-moderate"
    )
    batch_loader = _BatchLoader(
        data=load_data(model_data_stub), batch_size=1, iterations=1
    )

    for batch in batch_loader:
        out = client.classify(
            images=batch,
        )
        print(f"outputs:{out}")
