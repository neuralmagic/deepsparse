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
Generic Pydantic Schema for the Vision pipelines
If the pipeline is consuming images, its Input/Output
Schema should inherit from the Schemas below
"""

from typing import List, Union

import numpy
from pydantic import BaseModel, Field


__all__ = [
    "VisionInputSchema",
    "VisionOutputSchema",
]


class VisionInputSchema(BaseModel):
    """
    Input model for image classification
    """

    images: Union[str, List[str], numpy.ndarray, List[numpy.ndarray]] = Field(
        description="List of images to process"
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_files(cls, files: List[str], **kwargs) -> "VisionOutputSchema":
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


class VisionOutputSchema(BaseModel):
    pass
