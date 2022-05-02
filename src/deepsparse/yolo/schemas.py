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
Input/Output Schemas for Image Segmentation with YOLO
"""
from collections import namedtuple
from typing import List, Union

import numpy
from pydantic import BaseModel


__all__ = [
    "YOLOOutput",
    "YOLOInput",
]

_YOLOImageOutput = namedtuple(
    "_YOLOImageOutput", ["predictions", "boxes", "scores", "labels"]
)


class YOLOInput(BaseModel):
    """
    Input model for image classification
    """

    images: Union[str, List[numpy.ndarray], List[str]]

    class Config:
        arbitrary_types_allowed = True


class YOLOOutput(BaseModel):
    """
    Output model for image classification
    """

    predictions: List[List[List[float]]]
    boxes: List[List[List[float]]]
    scores: List[List[float]]
    labels: List[List[str]]

    def __getitem__(self, index):
        if index >= len(self.predictions):
            raise IndexError("Index out of range")

        return _YOLOImageOutput(
            self.predictions[index],
            self.boxes[index],
            self.scores[index],
            self.labels[index],
        )

    def __iter__(self):
        for index in range(len(self.predictions)):
            yield self[index]
