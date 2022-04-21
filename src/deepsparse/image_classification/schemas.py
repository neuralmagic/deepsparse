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

from typing import List, Union

import numpy
from pydantic import BaseModel


class ImageClassificationInput(BaseModel):
    """
    Input model for image classification
    """

    images: Union[str, List[numpy.ndarray], List[str]]

    class Config:
        arbitrary_types_allowed = True


class ImageClassificationOutput(BaseModel):
    """
    Input model for image classification
    """

    labels: List[Union[int, str]]
    scores: List[float]
