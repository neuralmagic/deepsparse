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

from typing import Any, List, Union

import numpy
from pydantic import BaseModel, Field


__all__ = [
    "YolactInputSchema",
    "YolactOutputSchema",
]


class YolactInputSchema(BaseModel):
    """
    Input Model for YOLACT
    """

    images: Union[str, List[str], List[Any], numpy.ndarray] = Field(
        description="List of images to process"
    )

    class Config:
        arbitrary_types_allowed = True


class YolactOutputSchema(BaseModel):
    """
    Output Model for YOLACT
    """
    classes: List[List[int]] = Field(description="List of predictions")
    scores: List[List[float]] = Field(
        description="List of bounding boxes, one for each prediction"
    )
    boxes: List[List[List[int]]] = Field(
        description="List of scores, one for each prediction"
    )
    masks: List[List[numpy.ndarray]] = Field(
        description="List of labels, one for each prediction"
    )

    class Config:
        arbitrary_types_allowed = True
