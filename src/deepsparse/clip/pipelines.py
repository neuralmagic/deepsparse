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

from pathlib import Path
from typing import Dict, List, Type

import numpy as np
import numpy.lingalg as la
import onnx
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from deepsparse.clip.constants import CLIP_RGB_MEANS, CLIP_RGB_STDS
from deepsparse.clip.schemas import (
    CLIPTextInput,
    CLIPTextOutput,
    CLIPVisualInput,
    CLIPVisualOutput,
    CLIPZeroShotInput,
    CLIPZeroShotOutput,
)
from deepsparse.pipeline import BasePipeline, Pipeline
from deepsparse.utils import model_to_path
from scipy.special import softmax


__all__ = ["CLIPVisualPipeline", "CLIPTextPipeline", "CLIPZeroShotPipeline"]


@Pipeline.register(task="clip_visual", default_model_path="")
class CLIPVisualPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._image_size = self._infer_image_size()
        self._preprocess_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=self._image_size,
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                ),
                transforms.CenterCrop(size=self._image_size),
            ]
        )
        # TODO: some argument to check if for CoCa? would have one additional output

    @property
    def input_schema(self) -> Type[CLIPVisualInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPVisualInput

    @property
    def output_schema(self) -> Type[CLIPVisualOutput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPVisualOutput

    def setup_onnx_file_path(self):
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """

        if Path(self.model_path).is_dir():
            return model_to_path(str(Path(self.model_path) / "model.onnx"))

        return model_to_path(self.model_path)

    # Should be the same for the captioning path? confirm
    def process_inputs(self, inputs: CLIPVisualInput) -> List[np.array]:
        """
        Preprocess inputs for CLIP's Visual Branch to comply with the DeepSparse Engine

        :param inputs: CLIPVisualInput
        :return: list of preprocessed numpy arrays
        """

        def _process_image(image) -> np.array:
            # TODO: handle the different input cases s
            image = self._preprocess_transforms(image)
            # convert to np.array to get channel dim (should we just use tensors?)
            # should make the image 8 bit
            image_array = np.array(image.convert("RGB"))
            # make channel dim the first dim
            image_array = image_array.transpose(2, 1, 0).astype("float")

            image_array /= 255.0
            image_array = (
                image_array - np.array(CLIP_RGB_MEANS).reshape((3, 1, 1))
            ) / np.array(CLIP_RGB_STDS.reshape((3, 1, 1)))

            image.close()

            return image_array

        # TODO: handle the different input cases (can be list, str or list of strings)
        image_batch = list(self.executor.map(self._process_image, inputs.images))
        return image_batch

    def process_engine_outputs(
        self, engine_outputs: List[np.array]
    ) -> CLIPVisualOutput:
        # TOD0: may or may not have tokens depending on if CoCa
        # For Visual Models (non-CoCa): output is batch * emebdding_dim
        # emgine_outputs = [batches] --> just return batch * embedding_dims? embeddings
        # if CoCa model: will have batch * dim * embedding_dim
        return self.output_schema()

    def _infer_image_size(self) -> int:
        """
        Infer and return the expected shape of the input tensor

        :return: The expected size of the input tensor for the onnx graph
        """
        onnx_model = onnx.load(self.onnx_file_path)
        input_tensor = onnx_model.graph.input[0]
        return input_tensor.type.tensor_type.shape.dim[2].dim_value


@Pipeline.register(task="clip_text", default_model_path="")
class CLIPTextPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = None  # Everything else would be stored in onnx class?

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
        # Can get_onnx_path_and_configs be used to fetch the tokenizer?
        if Path(self.model_path).is_dir():
            return model_to_path(str(Path(self.model_path) / "model.onnx"))

        return model_to_path(self.model_path)

    def process_inputs(self, inputs: CLIPTextInput) -> List[np.array]:
        """
        Preprocess inputs for CLIP's Trext Branch to comply with the DeepSparse Engine

        :param inputs: CLITextInput
        :return: list of preprocessed numpy arrays
        """
        # TODO: Handle the different input cases (for what inputs.inputs can be)
        # Other transformer pipelines handle an additional case: get the tokens and then
        # tokens_to_engine_input(tokens) is the final output?
        tokens = self.tokenizer(inputs.inputs)
        return [tokens]

    def process_engine_outputs(self, engine_outputs: List[np.array]) -> CLIPTextOutput:
        # TOD0: may or may not have tokens depending on if CoCa
        # For Text Models (non-CoCa): output is batch * emebdding_dim
        # emgine_outputs = [batches] --> just return batch * embedding_dims? embeddings
        # if CoCa model: will have batch * dim * embedding_dim
        return self.output_schema()


@BasePipeline.register(task="clip_zeroshot", default_model_path=None)
class CLIPZeroShotPipeline(BasePipeline):
    def __init__(self, visual_args: Dict, text_args: Dict, **kwargs):
        super().__init__(**kwargs)
        # Pass the same logger
        self.visual = Pipeline.create(task="clip_visual", **visual_args)
        self.text = Pipeline.create(task="clip_text", **text_args)

    def __call__(self, *args, **kwargs):
        # May have to override if inputs are different?
        pipeline_inputs = self.parse_inputs(*args, **kwargs)

        # Output processing for batches?

        if not isinstance(pipeline_inputs, self.input_schema):
            raise RuntimeError(
                f"Unable to parse {self.__class__} inputs into a "
                f"{self.input_schema} object. Inputs parsed to {type(pipeline_inputs)}"
            )

        visual_output = self.visual(pipeline_inputs.images)
        text_output = self.text(pipeline_inputs.text)

        visual_output /= la.norm(visual_output, axis=-1, keepdims=True)
        text_output /= la.norm(text_output, axis=-1, keepdims=True)

        output_product = 100.0 * visual_output @ text_output.T
        text_probs = softmax(output_product, axis=-1)
        return text_probs

    def input_schema(self) -> Type[CLIPZeroShotInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPZeroShotInput

    def output_schema(self) -> Type[CLIPZeroShotOutput]:
        """
        :return: pydantic model class that outputs to this pipeline must comply to
        """
        return CLIPZeroShotOutput
