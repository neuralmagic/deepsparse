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
Input/Output Schemas for Object Detection with YOLO
"""

from collections import namedtuple
from typing import List

from pydantic import BaseModel, Field

from deepsparse.pipelines.computer_vision import ComputerVisionSchema


__all__ = [
    "YOLOOutput",
    "YOLOInput",
]

_YOLOImageOutput = namedtuple("_YOLOImageOutput", ["boxes", "scores", "labels"])


class YOLOInput(ComputerVisionSchema):
    """
    Input model for object detection
    """

    iou_thres: float = Field(
        default=0.25,
        description="minimum IoU overlap threshold for a prediction to be valid",
    )
    conf_thres: float = Field(
        default=0.45,
        description="minimum confidence score for a prediction to be valid",
    )
    multi_label: bool = Field(
        default=False,
        description=(
            "when true, allow multi-label assignment to each detected object. Defaults "
            "to False to mimic yolov5 detection pathway. Note that yolov5 validation "
            "pathway by default run with multi_label on"
        ),
    )


class YOLOOutput(BaseModel):
    """
    Output model for object detection
    """

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
        if index >= len(self.boxes):
            raise IndexError("Index out of range")

        return _YOLOImageOutput(
            self.boxes[index],
            self.scores[index],
            self.labels[index],
        )

    def __iter__(self):
        for index in range(len(self.boxes)):
            yield self[index]
