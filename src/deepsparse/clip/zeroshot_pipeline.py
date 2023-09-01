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

from typing import Any, List, Type

import numpy as np
from numpy import linalg as lingalg
from pydantic import BaseModel, Field

from deepsparse.clip import CLIPTextInput, CLIPVisualInput
from deepsparse.pipeline import BasePipeline, Pipeline
from scipy.special import softmax


__all__ = ["CLIPZeroShotInput", "CLIPZeroShotOutput", "CLIPZeroShotPipeline"]


class CLIPZeroShotInput(BaseModel):
    """
    Input for the CLIP Zero Shot Model
    """

    image: CLIPVisualInput = Field(
        description="Image(s) to run zero-shot prediction. See CLIPVisualPipeline "
        "for details."
    )
    text: CLIPTextInput = Field(
        description="Text/classes to run zero-shot prediction "
        "see CLIPTextPipeline for details."
    )


class CLIPZeroShotOutput(BaseModel):
    """
    Output for the CLIP Zero Shot Model
    """

    text_scores: List[Any] = Field(
        description="np.array consisting of probabilities " " each class provided."
    )


@BasePipeline.register(task="clip_zeroshot", default_model_path=None)
class CLIPZeroShotPipeline(BasePipeline):
    """
    Pipeline designed to run zero-shot classification given a list of images and
    possible classes. The CLIPZeroShotPipeline relies on two pipelines, the
    CLIPTextPipeline which handles CLIP's text branch adn the CLIPVisualPipeline
    which handles CLIP's visual branch. The final score calculations are handled and
    returned by the CLIPZeroShotPipeline. See README.md for a detailed example.

    :param visual_model_path: either a local path or sparsezoo stub for the CLIP visual
    branch onnx model
    :param text_model_path: either a local path or sparsezoo stub for the CLIP text
    branch onnx model
    :param pipeline_engine_args: dictionary of arguments specific to running the
    pipeline engine. Applied to both the text branch pipeline and visual branch pipeline
    See pipeline.py for a full list of arguments.

    """

    def __init__(
        self,
        visual_model_path: str,
        text_model_path: str,
        pipeline_engine_args: dict = None,
        **kwargs,
    ):
        if pipeline_engine_args is None:
            pipeline_engine_args = {}

        pipeline_engine_args["model_path"] = visual_model_path
        self.visual = Pipeline.create(task="clip_visual", **pipeline_engine_args)
        pipeline_engine_args["model_path"] = text_model_path
        self.text = Pipeline.create(task="clip_text", **pipeline_engine_args)

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        pipeline_inputs = self.parse_inputs(*args, **kwargs)

        if not isinstance(pipeline_inputs, self.input_schema):
            raise RuntimeError(
                f"Unable to parse {self.__class__} inputs into a "
                f"{self.input_schema} object. Inputs parsed to {type(pipeline_inputs)}"
            )

        visual_output = self.visual(pipeline_inputs.image).image_embeddings[0]
        text_output = self.text(pipeline_inputs.text).text_embeddings[0]

        visual_output /= lingalg.norm(visual_output, axis=-1, keepdims=True)
        text_output /= lingalg.norm(text_output, axis=-1, keepdims=True)

        output_product = 100.0 * visual_output @ text_output.T
        text_probs = softmax(output_product, axis=-1)

        return self.output_schema(text_scores=np.vsplit(text_probs, len(text_probs)))

    @property
    def input_schema(self) -> Type[CLIPZeroShotInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPZeroShotInput

    @property
    def output_schema(self) -> Type[CLIPZeroShotOutput]:
        """
        :return: pydantic model class that outputs to this pipeline must comply to
        """
        return CLIPZeroShotOutput
