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
from typing import Any, List, Union

from pydantic import BaseModel, Field


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

    images: Union[str, List[str], List[Any]] = Field(
        description="List of Images to process"
    )
    iou_thres: float = Field(
        default=0.25,
        description="minimum IoU overlap threshold for a prediction to be valid",
    )
    conf_thres: float = Field(
        default=0.45,
        description="minimum confidence score for a prediction to be valid",
    )

    @classmethod
    def from_files(cls, files: List[str], **kwargs) -> "YOLOInput":
        """
        :param files: list of file paths to create YOLOInput from
        :param kwargs: extra keyword args to pass to YOLOInput constructor
        :return: YOLOInput constructed from files
        """
        if "images" in kwargs:
            raise ValueError(
                f"argument 'images' cannot be specified in {cls.__name__} when "
                "constructing from file(s)"
            )
        return cls(images=files, **kwargs)

    class Config:
        arbitrary_types_allowed = True


class YOLOOutput(BaseModel):
    """
    Output model for image classification
    """

    predictions: List[List[List[float]]] = Field(description="List of predictions")
    boxes: List[List[List[float]]] = Field(
        description="List of bounding boxes, one for each prediction"
    )
    scores: List[List[float]] = Field(
        description="List of scores, one for each prediction"
    )
    labels: List[List[str]] = Field(
        description="List of labels, one for each prediction"
    )

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
