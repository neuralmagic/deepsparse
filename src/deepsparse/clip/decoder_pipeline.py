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
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.utils import model_to_path


__all__ = ["CLIPDecoderInput", "CLIPDecoderOutput", "CLIPDecoderPipeline"]


class CLIPDecoderInput(BaseModel):
    """
    Input for the CLIP Decoder Branch
    """

    text_embeddings: Any = Field(
        description="np.array of text emebddings from the " "text branch"
    )
    image_embeddings: Any = Field(
        description="np.array of image embeddings from the " "visual branch"
    )


class CLIPDecoderOutput(BaseModel):
    """
    Output for the CLIP Decoder Branch
    """

    logits: List[Any] = Field(
        description="np.array of logits produced from the decoder."
    )


@Pipeline.register(task="clip_decoder", default_model_path=None)
class CLIPDecoderPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[CLIPDecoderInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPDecoderInput

    @property
    def output_schema(self) -> Type[CLIPDecoderOutput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPDecoderOutput

    def setup_onnx_file_path(self):
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        return model_to_path(self.model_path)

    def process_inputs(self, inputs: CLIPDecoderInput) -> List[np.array]:
        """
        Preprocess inputs for CLIP's Decoder Branch to comply with the DeepSparse Engine

        :param inputs: CLIPDecoderInput
        :return: list of preprocessed numpy arrays
        """
        image_embeddings = inputs.image_embeddings
        text_embeddings = inputs.text_embeddings
        return [image_embeddings, text_embeddings]

    def process_engine_outputs(
        self, engine_outputs: List[np.array]
    ) -> CLIPDecoderOutput:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        return self.output_schema(logits=engine_outputs)
