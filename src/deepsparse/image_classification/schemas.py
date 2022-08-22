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
from typing import Generator, Iterable, List, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse.pipelines import Joinable, Splittable
from deepsparse.pipelines.computer_vision import ComputerVisionSchema


__all__ = [
    "ImageClassificationInput",
    "ImageClassificationOutput",
]


class ImageClassificationInput(ComputerVisionSchema, Splittable):
    """
    Input model for image classification
    """

    def split(self) -> Generator["ImageClassificationInput", None, None]:
        """
        Split a current `ImageClassificationInput` object with a batch size b, into a
        generator of b smaller objects with batch size 1, the returned
        object can be iterated on.

        :return: A Generator of smaller `ImageClassificationInput` objects each
            representing an input of batch-size 1
        """

        images = self.images

        is_batch_size_one = isinstance(images, str) or (
            isinstance(images, numpy.ndarray) and images.ndim == 3
        )
        if is_batch_size_one:
            # case 1: str, numpy.ndarray(3D)
            yield self

        elif isinstance(images, numpy.ndarray) and images.ndim != 4:
            raise ValueError(f"Could not breakdown {self} into smaller batches")

        else:
            # case 2: List[str, Any], numpy.ndarray(4D) -> multiple images of size 1
            for image in images:
                yield ImageClassificationInput(images=image)


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
        Takes in ab Iterable of `ImageClassificationOutput` objects and combines
        them into one object representing a bigger batch size

        :return: A new `ImageClassificationOutput` object that represents a bigger batch
        """
        if len(outputs) == 1:
            return outputs[0]

        scores = list()
        labels = list()

        for output in outputs:
            labels.append(output.labels)
            scores.append(output.scores)

        return ImageClassificationOutput(scores=scores, labels=labels)
