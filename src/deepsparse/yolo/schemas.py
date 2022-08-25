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
from typing import Generator, Iterable, List

import numpy
from pydantic import BaseModel, Field

from deepsparse.pipelines import Joinable, Splittable
from deepsparse.pipelines.computer_vision import ComputerVisionSchema


__all__ = [
    "YOLOOutput",
    "YOLOInput",
]

_YOLOImageOutput = namedtuple(
    "_YOLOImageOutput", ["predictions", "boxes", "scores", "labels"]
)


class YOLOInput(ComputerVisionSchema, Splittable):
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

    def split(self) -> Generator["YOLOInput", None, None]:
        """
        Split a current `YOLOInput` object with a batch size b, into a
        generator of b smaller objects with batch size 1, the returned
        object can be iterated on.

        :return: A Generator of smaller `YOLOInput` objects each
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
                yield YOLOInput(images=image)


class YOLOOutput(BaseModel, Joinable):
    """
    Output model for object detection
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

    @staticmethod
    def join(
        outputs: Iterable["YOLOOutput"],
    ) -> "YOLOOutput":
        """
        Takes in ab Iterable of `YOLOOutput` objects and combines
        them into one object representing a bigger batch size

        :return: A new `YOLOOutput` object that represents a bigger batch
        """
        predictions = list()
        boxes = list()
        scores = list()
        labels = list()

        for yolo_output in outputs:
            for image_output in yolo_output:
                predictions.append(image_output.predictions)
                boxes.append(image_output.boxes)
                scores.append(image_output.scores)
                labels.append(image_output.labels)

        return YOLOOutput(
            predictions=predictions, boxes=boxes, scores=scores, labels=labels
        )
