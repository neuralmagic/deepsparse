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

from typing import Any, List, Type, Union

import numpy as np
from pydantic import BaseModel, Field

from deepsparse.pipeline import Pipeline
from deepsparse.utils import model_to_path
from open_clip.tokenizer import tokenize


__all__ = ["CLIPTextInput", "CLIPTextOutput", "CLIPTextPipeline"]


class CLIPTextInput(BaseModel):
    """
    Input for the CLIP Text Branch
    """

    text: Union[str, List[str], Any, List[Any]] = Field(
        description="Either raw strings or an np.array with tokenized text"
    )


class CLIPTextOutput(BaseModel):
    """
    Output for the CLIP Text Branch
    """

    text_embeddings: List[Any] = Field(
        description="np.array of text embeddings. For the caption "
        "pipeline, a list of two embeddings is produced. For zero-shot "
        "classifcation, one array is produced with the embeddings stacked along "
        "batch axis."
    )


@Pipeline.register(task="clip_text", default_model_path=None)
class CLIPTextPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenize

    @property
    def input_schema(self) -> Type[CLIPTextInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPTextInput

    @property
    def output_schema(self) -> Type[CLIPTextOutput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPTextOutput

    def setup_onnx_file_path(self):
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        return model_to_path(self.model_path)

    def process_inputs(self, inputs: CLIPTextInput) -> List[np.ndarray]:
        """
        Preprocess inputs for CLIP's Trext Branch to comply with the DeepSparse Engine

        :param inputs: CLITextInput
        :return: list of preprocessed numpy arrays
        """
        if not isinstance(inputs.text, list):
            inputs.text = [inputs.text]

        # If passing in an array, part of the captioning pipeline. No need to tokenize
        if not isinstance(inputs.text[0], str):
            return inputs.text

        tokens = self.tokenizer(inputs.text)
        tokens = [np.array(t).astype(np.int32) for t in tokens]
        tokens = np.stack(tokens, axis=0)
        return [tokens]

    def process_engine_outputs(
        self, engine_outputs: List[np.array], **kwargs
    ) -> CLIPTextOutput:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        return self.output_schema(text_embeddings=engine_outputs)
