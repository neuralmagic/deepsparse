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
from typing import Any, Iterable, List, Optional, TextIO

import numpy
from PIL import Image
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
    return_masks: bool = Field(
        default=True,
        description="Controls whether the pipeline should additionally "
        "return segmentation masks (if running a segmentation model)",
    )
    return_intermediate_outputs: bool = Field(
        default=False,
        description="Controls whether the pipeline should additionally "
        "return intermediate outputs from the model",
    )

    @classmethod
    def from_files(
        cls, files: Iterable[TextIO], *args, from_server: bool = False, **kwargs
    ) -> "YOLOInput":
        """
        :param files: Iterable of file pointers to create YOLOInput from
        :param kwargs: extra keyword args to pass to YOLOInput constructor
        :return: YOLOInput constructed from files
        """
        if "images" in kwargs:
            raise ValueError(
                f"argument 'images' cannot be specified in {cls.__name__} when "
                "constructing from file(s)"
            )
        files_numpy = [numpy.array(Image.open(file)) for file in files]
        input_schema = cls(
            # if the input comes through the client-server communication
            # do not return segmentation masks
            *args,
            images=files_numpy,
            return_masks=not from_server,
            **kwargs,
        )
        return input_schema

    class Config:
        arbitrary_types_allowed = True


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
    intermediate_outputs: Optional[Any] = Field(
        default=None,
        description="Intermediate outputs from the YOLOv8 segmentation model.",
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
