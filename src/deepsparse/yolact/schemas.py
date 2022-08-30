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
from typing import Any, Iterable, List, Optional, TextIO, Union

import numpy
from PIL import Image
from pydantic import BaseModel, Field

from deepsparse.pipelines.computer_vision import ComputerVisionSchema


__all__ = [
    "YOLACTInputSchema",
    "YOLACTOutputSchema",
]

_YOLACTImageOutput = namedtuple(
    "_YOLACTImageOutput", ["classes", "scores", "boxes", "masks"]
)


class YOLACTInputSchema(ComputerVisionSchema):
    """
    Input Model for YOLACT
    """

    confidence_threshold: float = Field(
        default=0.05,
        description="Confidence threshold applied to the raw detection at "
        "`detection` step. If a raw detection's score is lower "
        "than the threshold, it will be automatically discarded",
    )
    nms_threshold: float = Field(
        default=0.5,
        description="Minimum IoU overlap threshold for a prediction to "
        "consider it valid (used in Non-Maximum-Suppression step)",
    )
    top_k_preprocessing: int = Field(
        default=200,
        description="The maximal number of best detections (per class) to be "
        "kept after the Non-Maximum-Suppression step",
    )
    max_num_detections: int = Field(
        default=100,
        description="The maximal number of best detections (across all classes) "
        "to be kept after the Non-Maximum-Suppression step",
    )
    score_threshold: float = Field(
        default=0.0,
        description="Confidence threshold applied to the raw detection at "
        "`postprocess` step (optional)",
    )
    return_masks: bool = Field(
        default=True,
        description="Controls whether the pipeline should additionally "
        "return segmentation masks",
    )

    @classmethod
    def from_files(
        cls, files: Iterable[TextIO], *args, from_server: bool = False, **kwargs
    ) -> "YOLACTInputSchema":
        """
        :param files: Iterable of file pointers to create YOLACTInput from
        :param kwargs: extra keyword args to pass to YOLACTInput constructor
        :return: YOLACTInput constructed from files
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


class YOLACTOutputSchema(BaseModel):
    """
    Output Model for YOLACT
    """

    classes: List[List[Optional[Union[int, str]]]] = Field(
        description="List of predictions"
    )
    scores: List[List[Optional[float]]] = Field(
        description="List of scores, one for each prediction"
    )
    boxes: List[List[Optional[List[float]]]] = Field(
        description="List of bounding boxes, one for each prediction"
    )
    masks: Optional[List[Any]] = Field(
        description="List of masks, one for each prediction"
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
            self.masks[index] if self.masks is not None else None,
        )

    def __iter__(self):
        for index in range(len(self.classes)):
            yield self[index]
