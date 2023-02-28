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

from typing import Any, List

from pydantic import BaseModel, Field


__all__ = ["YOLOSegOutput"]


class YOLOSegOutput(BaseModel):
    """
    Output model for YOLOv8 Segmentation model
    """

    boxes: List[List[List[float]]] = Field(
        description="List of bounding boxes, one for each prediction"
    )
    scores: List[List[float]] = Field(
        description="List of scores, one for each prediction"
    )
    classes: List[List[str]] = Field(
        description="List of labels, one for each prediction"
    )
    masks: List[Any] = Field(description="List of masks, one for each prediction")
