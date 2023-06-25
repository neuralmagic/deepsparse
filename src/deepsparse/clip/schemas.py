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


from pydantic import BaseModel

from deepsparse.pipelines.computer_vision import ComputerVisionSchema


__all__ = [
    "CLIPVisualInput",
    "CLIPVisualOutput",
    "CLIPTextInput",
    "CLIPTextOutput",
    "CLIPZeroShotInput",
    "CLIPZeroShotOutput",
]


class CLIPVisualInput(ComputerVisionSchema):
    """
    Input for CLIP Visual Branch
    """


class CLIPVisualOutput(BaseModel):
    """
    Output for CLIP Visual Branch
    """


class CLIPTextInput(BaseModel):
    """
    Input for the CLIP Text Branch
    - Should include inputs (passed through tokenizer)
    """


class CLIPTextOutput(BaseModel):
    """
    Output for the CLIP Text Branch
    """


class CLIPZeroShotInput(BaseModel):
    """
    - images
    - text
    Input for the CLIP Zero Shot Model
    """


class CLIPZeroShotOutput(BaseModel):
    """
    Output for the CLIP Zero Shot Model
    - text_scores
    """
