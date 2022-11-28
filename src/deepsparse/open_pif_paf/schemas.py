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

import numpy
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

    cif: numpy.ndarray = Field(
        description="CIF field with 17 x 5 channels "
        "(resulting array has dimensions: (B,17,5,13,17))"
    )
    caf: numpy.ndarray = Field(
        description="CIF field with 19 x 8 channels "
        "(resulting array has dimensions: (B,19,8,13,17))"
    )

    class Config:
        arbitrary_types_allowed = True
