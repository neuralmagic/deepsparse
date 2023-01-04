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

from typing import List, Tuple

from pydantic import BaseModel, Field

from deepsparse.pipelines.computer_vision import ComputerVisionSchema


__all__ = [
    "OpenPifPafInput",
    "OpenPifPafOutput",
]


class OpenPifPafInput(ComputerVisionSchema):
    """
    Input model for Open Pif Paf
    """

    pass


class OpenPifPafOutput(BaseModel):
    """
    Output model for Open Pif Paf
    """

    data: List[List[List[List[float]]]] = Field(
        description="List of list-formatted arrays "
        "(one array per prediction) of shape (N, 3) "
        "where N is the number of keypoints (joints). "
        "Each array contains the x coordinate, y coordinate, "
        "and confidence values for each joint."
    )
    keypoints: List[List[List[str]]] = Field(
        description="List of names of skelethon joints, "
        "one list for each prediction."
    )
    scores: List[List[float]] = Field(
        description="List of pose (skelethon) detection probabilities, "
        "one value for each prediction."
    )
    skeletons: List[List[List[Tuple[int, int]]]] = Field(
        description="List of skelethon body part connections. "
        "For every prediction, it is a list of tuples of body "
        "part indices. "
    )

    class Config:
        arbitrary_types_allowed = True
