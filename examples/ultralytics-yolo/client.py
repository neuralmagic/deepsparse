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
Client object for making requests to the example DeepSparse YOLO inference server
"""


import time
from typing import List, Union

import numpy
import requests

from deepsparse.utils import arrays_to_bytes, bytes_to_arrays
from deepsparse_utils import load_image


class YoloDetectionClient:
    """
    Client object for making requests to the example DeepSparse Yolo inference server

    :param address: IP address of the server, default is 0.0.0.0
    :param port: Port the server is hosted on, default is 5543
    """

    def __init__(self, address: str = "0.0.0.0", port: str = 5543):
        self._url = f"http://{address}:{port}/predict"

    def detect(
        self, images: Union[str, numpy.ndarray, List[str], List[numpy.ndarray]]
    ) -> List[numpy.ndarray]:
        """
        :param images: list of or singular file paths or numpy arrays of images to
            run the detection model on. Number of images should equal model batch size
        :return: list of post-processed object detection results for each image
            including class label, likelihood, and bounding box coordinates
        """
        if not isinstance(images, List):
            images = [images]
        images = numpy.stack([load_image(image)[0] for image in images])

        print(f"Sending batch of {len(images)} images for detection to {self._url}")

        start = time.time()
        # Encode inputs
        data = arrays_to_bytes([images])
        # Send data to server for inference
        response = requests.post(self._url, data=data)
        # Decode outputs
        outputs = bytes_to_arrays(response.content)
        total_time = time.time() - start
        print(f"Round-trip time took {total_time * 1000.0:.4f}ms")

        return outputs
