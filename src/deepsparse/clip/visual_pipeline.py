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
import onnx
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from deepsparse.clip.constants import CLIP_RGB_MEANS, CLIP_RGB_STDS
from deepsparse.pipeline import Pipeline
from deepsparse.pipelines.computer_vision import ComputerVisionSchema
from deepsparse.utils import model_to_path


__all__ = ["CLIPVisualInput", "CLIPVisualOutput", "CLIPVisualPipeline"]


class CLIPVisualInput(ComputerVisionSchema):
    """
    Input for CLIP Visual Branch

    """


class CLIPVisualOutput(BaseModel):
    """
    Output for CLIP Visual Branch

    """

    image_embeddings: List[Any] = Field(
        description="np.arrays consisting of image embeddings. For the caption "
        "pipeline, a list of two image embeddings is produced. For zero-shot "
        "classifcation, one array is produced with the embeddings stacked along "
        "batch axis."
    )


@Pipeline.register(task="clip_visual", default_model_path=None)
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
                    antialias="warn",
                ),
                transforms.CenterCrop(size=(self._image_size, self._image_size)),
            ]
        )

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
        return model_to_path(self.model_path)

    def process_inputs(self, inputs: CLIPVisualInput) -> List[np.array]:
        """
        Preprocess inputs for CLIP's Visual Branch to comply with the DeepSparse Engine

        :param inputs: CLIPVisualInput
        :return: list of preprocessed numpy arrays
        """
        if not isinstance(inputs.images, list):
            inputs.images = [inputs.images]

        def _process_image(image) -> np.array:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
                image = self._preprocess_transforms(image)

                image_array = np.array(image)

                # make channel dim the first dim
                image_array = image_array.transpose(2, 0, 1).astype("float32")

                image_array /= 255.0
                image_array = (
                    image_array - np.array(CLIP_RGB_MEANS).reshape((3, 1, 1))
                ) / np.array(CLIP_RGB_STDS).reshape((3, 1, 1))

                image.close()
            else:
                image_array = image
            return np.ascontiguousarray(image_array, dtype=np.float32)

        batch = list(self.executor.map(_process_image, inputs.images))
        if batch[0].ndim == 3:
            batch = [np.stack(batch, axis=0)]
        return batch

    def process_engine_outputs(
        self, engine_outputs: List[np.array]
    ) -> CLIPVisualOutput:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        return self.output_schema(image_embeddings=engine_outputs)

    def _infer_image_size(self) -> int:
        """
        Infer and return the expected shape of the input tensor

        :return: The expected size of the input tensor for the onnx graph
        """
        onnx_model = onnx.load(self.onnx_file_path, load_external_data=False)
        input_tensor = onnx_model.graph.input[0]
        return input_tensor.type.tensor_type.shape.dim[2].dim_value
