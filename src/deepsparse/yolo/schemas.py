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
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Union

from pydantic import BaseModel, Field

from deepsparse.utils import Joinable, Splittable


__all__ = [
    "YOLOOutput",
    "YOLOInput",
]

_YOLOImageOutput = namedtuple(
    "_YOLOImageOutput", ["predictions", "boxes", "scores", "labels"]
)


class YOLOInput(BaseModel, Splittable):
    """
    Input model for object detection
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

    @staticmethod
    def create_test_inputs(
        batch_size: int = 1,
    ) -> Dict[str, Union[Union[List[str], str, List[Any]]]]:
        """
        Create and return a test input for this schema

        :param batch_size: The batch_size of inputs to return
        :return: A dict representing inputs for Yolo pipeline
        """

        sample_image_path = Path(__file__).parents[0] / "sample_images" / "basilica.jpg"
        sample_image_abs_path = str(sample_image_path.absolute())

        images = [sample_image_abs_path for _ in range(batch_size)]
        return {"images": images}

    def split(self) -> Generator["YOLOInput", None, None]:
        """
        Split a current `YOLOInput` object with a batch size b, into a
        generator of b smaller objects with batch size 1, the returned
        object can be iterated on.

        :return: A Generator of smaller `YOLOInput` objects each
            representing an input of batch-size 1
        """

        images = self.images

        # case 1: do nothing if single input of batch_size 1
        if isinstance(images, str):
            yield self

        elif isinstance(images, list) and len(images) and isinstance(images[0], str):
            # case 2: List[str, Any] -> multiple images of size 1
            for image in images:
                yield YOLOInput(
                    images=image,
                )

        else:
            raise ValueError(f"Could not breakdown {self} into smaller batches")


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
            predictions=predictions,
            boxes=boxes,
            scores=scores,
            labels=labels,
        )
