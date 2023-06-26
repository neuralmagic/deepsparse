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

from typing import Any, List, Union

from pydantic import BaseModel, Field

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

    image_embeddings: List[Any] = Field(
        description="Image embeddings for the single image or list of embeddings for "
        "multiple images"
    )


class CLIPTextInput(BaseModel):
    """
    Input for the CLIP Text Branch
    """

    text: Union[str, List[str]] = Field(description="List of text to process")


class CLIPTextOutput(BaseModel):
    """
    Output for the CLIP Text Branch
    """

    text_embeddings: List[Any] = Field(
        description="Text embeddings for the single text or list of embeddings for "
        "multiple."
    )


class CLIPZeroShotInput(BaseModel):
    """
    Input for the CLIP Zero Shot Model
    """

    image: str = Field(description="Path to image to run zero-shot prediction on.")
    text: Union[str, List[str]] = Field(description="List of text to process")


class CLIPZeroShotOutput(BaseModel):
    """
    Output for the CLIP Zero Shot Model
    """

    # Maybe change this to a dictionary? where keys are text inputs
    text_scores: List[float] = Field(description="Probability of each text class")
