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
Input/Output Schemas for Image Classification.
"""
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse.pipelines import Joinable, Splittable


__all__ = [
    "ImageClassificationInput",
    "ImageClassificationOutput",
]


class ImageClassificationInput(BaseModel, Splittable):
    """
    Input model for image classification
    """

    images: Union[str, List[str], List[Any], numpy.ndarray] = Field(
        description="List of Images to process"
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_files(cls, files: List[str], **kwargs) -> "ImageClassificationInput":
        """
        :param files: list of file paths to create ImageClassificationInput from
        :return: ImageClassificationInput constructed from files
        """
        if kwargs:
            raise ValueError(
                f"{cls.__name__} does not support additional arguments "
                f"{list(kwargs.keys())}"
            )
        return cls(images=files)

    @staticmethod
    def create_test_inputs(
        batch_size: int = 1,
    ) -> Dict[str, Union[Union[List[str], str, List[Any]]]]:
        """
        Create and return a test input for this schema

        :param batch_size: The batch_size of inputs to return
        :return: A dict representing inputs for Image Classification pipeline
        """

        sample_image_path = Path(__file__).parents[0] / "sample_images" / "basilica.jpg"
        sample_image_abs_path = str(sample_image_path.absolute())

        images = [sample_image_abs_path for _ in range(batch_size)]
        return {"images": images}

    def split(self) -> Generator["ImageClassificationInput", None, None]:
        """
        Split a current `ImageClassificationInput` object with a batch size b, into a
        generator of b smaller objects with batch size 1, the returned
        object can be iterated on.

        :return: A Generator of smaller `ImageClassificationInput` objects each
            representing an input of batch-size 1
        """

        images = self.images

        # case 1: do nothing if single input of batch_size 1
        if isinstance(images, str):
            yield self

        elif isinstance(images, list) and len(images) and isinstance(images[0], str):
            # case 2: List[str] -> multiple images of size 1
            for image in images:
                yield ImageClassificationInput(
                    images=image,
                )
        else:
            raise ValueError(f"Could not breakdown {self} into smaller batches")


class ImageClassificationOutput(BaseModel, Joinable):
    """
    Output model for image classification
    """

    labels: List[Union[int, str, List[int], List[str]]] = Field(
        description="List of labels, one for each prediction"
    )
    scores: List[Union[float, List[float]]] = Field(
        description="List of scores, one for each prediction"
    )

    @staticmethod
    def join(
        outputs: Iterable["ImageClassificationOutput"],
    ) -> "ImageClassificationOutput":
        """
        Takes in ab Iterable of `YOLOOutput` objects and combines
        them into one object representing a bigger batch size

        :return: A new `YOLOOutput` object that represents a bigger batch
        """
        if len(outputs) == 1:
            return outputs[0]

        scores = list()
        labels = list()

        for output in outputs:
            labels.append(output.labels)
            scores.append(output.scores)

        return ImageClassificationOutput(
            scores=scores,
            labels=labels,
        )
