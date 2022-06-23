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
Input/Output Schemas for Image Segmentation with YOLACT
"""

from collections import namedtuple
from typing import List, Union

import numpy
from pydantic import BaseModel, Field


__all__ = [
    "YOLACTInputSchema",
    "YOLACTOutputSchema",
]

_YOLACTImageOutput = namedtuple(
    "_YOLACTImageOutput", ["classes", "scores", "boxes", "masks"]
)


class YOLACTInputSchema(BaseModel):
    """
    Input Model for YOLACT
    """

    images: Union[str, numpy.ndarray, List[Union[str, numpy.ndarray]]] = Field(
        description="List of images to process"
    )

    confidence_threshold: float = Field(default = 0.05, description ="Confidence threshold applied to the raw detection at `detection` step. If a raw detection's score lower than threshold, it will be automatically discarded")
    nms_threshold: float = Field(default=0.5, description="Minimum IoU overlap threshold for a prediction to be valid (used in Non-Maximum-Suppression step)")
    top_k_preprocessing: int = Field(default=200, description="The limiting number of best detections (per class) to be kept after the Non-Maximum-Suppression step")
    max_num_detections: int = Field(default=100, description="The limiting number of best detections (across all classes) to be kept after the Non-Maximum-Suppression step")
    score_threshold: float = Field(default=0.0, description="Confidence threshold applied to the raw detection at `postprocess` step (optional)")


    @classmethod
    def from_files(cls, files: List[str], **kwargs) -> "YOLACTInputSchema":
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


class YOLACTOutputSchema(BaseModel):
    """
    Output Model for YOLACT
    """

    classes: List[List[int]] = Field(description="List of predictions")
    scores: List[List[float]] = Field(
        description="List of bounding boxes, one for each prediction"
    )
    boxes: List[List[List[float]]] = Field(
        description="List of scores, one for each prediction"
    )
    masks: List[List[numpy.ndarray]] = Field(
        description="List of labels, one for each prediction"
    )

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, index):
        if index >= len(self.classes):
            raise IndexError("Index out of range")

        return _YOLACTImageOutput(
            self.classes[index],
            self.scores[index],
            self.boxes[index],
            self.masks[index],
        )

    def __iter__(self):
        for index in range(len(self.classes)):
            yield self[index]
