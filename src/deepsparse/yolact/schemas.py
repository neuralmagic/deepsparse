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
from typing import Generator, Iterable, List, Optional, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse.pipelines import Joinable, Splittable
from deepsparse.pipelines.computer_vision import ComputerVisionSchema


__all__ = [
    "YOLACTInputSchema",
    "YOLACTOutputSchema",
]

_YOLACTImageOutput = namedtuple(
    "_YOLACTImageOutput", ["classes", "scores", "boxes", "masks"]
)


class YOLACTInputSchema(ComputerVisionSchema, Splittable):
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

    class Config:
        arbitrary_types_allowed = True

    def split(self) -> Generator["YOLACTInputSchema", None, None]:
        """
        Split a current `YOLACTInputSchema` object with a batch size b, into a
        generator of b smaller objects with batch size 1, the returned
        object can be iterated on.

        :return: A Generator of smaller `YOLACTInputSchema` objects each
            representing an input of batch-size 1
        """
        images = self.images

        is_batch_size_1 = isinstance(images, str) or (
            isinstance(images, numpy.ndarray) and images.ndim == 3
        )
        if is_batch_size_1:
            # case 1: str, numpy.ndarray(3D)
            yield self

        elif isinstance(images, numpy.ndarray) and images.ndim != 4:
            raise ValueError(f"Could not breakdown {self} into smaller batches")

        else:
            # case 2: List[str, Any], numpy.ndarray(4D) -> multiple images of size 1
            for image in images:
                yield YOLACTInputSchema(images=image)


class YOLACTOutputSchema(BaseModel, Joinable):
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
    masks: List[Optional[List[float]]] = Field(
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
            self.masks[index],
        )

    def __iter__(self):
        for index in range(len(self.classes)):
            yield self[index]

    @staticmethod
    def join(outputs: Iterable["YOLACTOutputSchema"]) -> "YOLACTOutputSchema":
        """
        Takes in ab Iterable of `YOLACTOutputSchema` objects and combines
        them into one object representing a bigger batch size

        :return: A new `YOLACTOutputSchema` object that represents a bigger batch
        """

        classes = list()
        scores = list()
        boxes = list()
        masks = list()

        for yolact_output in outputs:
            for image_output in yolact_output:
                classes.append(image_output.classes)
                scores.append(image_output.scores)
                boxes.append(image_output.boxes)
                masks.append(image_output.masks)

        return YOLACTOutputSchema(
            classes=classes, scores=scores, boxes=boxes, masks=masks
        )
