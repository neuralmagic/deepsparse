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

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field


__all__ = ["YOLOSegOutput"]


class YOLOSegOutput(BaseModel):
    """
    Output model for YOLOv8 Segmentation model
    """

    boxes: List[List[Optional[List[float]]]] = Field(
        description="List of bounding boxes, one for each prediction"
    )
    scores: List[List[Optional[float]]] = Field(
        description="List of scores, one for each prediction"
    )
    classes: List[List[Optional[str]]] = Field(
        description="List of labels, one for each prediction"
    )
    masks: Optional[List[Any]] = Field(
        description="List of masks, one for each prediction"
    )

    intermediate_outputs: Optional[Tuple[Any, Any]] = Field(
        default=None,
        description="A tuple that contains of intermediate outputs "
        "from the YOLOv8 segmentation model. The tuple"
        "contains two items: predictions from the model"
        "and mask prototypes",
    )
