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

from typing import Dict, List, Optional, Tuple

import numpy
import onnx
from PIL import Image
from torchvision import transforms

from deepsparse.image_classification.constants import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from deepsparse.pipelines.computer_vision import ComputerVisionSchema
from deepsparse.v2.operators import Operator


class ImageClassificationInput(ComputerVisionSchema):
    """
    Input model for image classification
    """


__all__ = ["ImageClassificationPreProcess"]


class ImageClassificationPreProcess(Operator):
    """
    Image Classification pre-processing operator. This Operator is expected to process
    the user inputs and prepare them for the engine. Inputs to this Operator are
    expected to follow the ImageClassificationInput schema.
    """

    input_schema = ImageClassificationInput
    output_schema = None

    def __init__(self, model_path: str, image_size: Optional[Tuple[int]] = None):
        self.model_path = model_path
        self._image_size = image_size or self._infer_image_size()
        non_rand_resize_scale = 256.0 / 224.0  # standard used
        self._pre_normalization_transforms = transforms.Compose(
            [
                transforms.Resize(
                    tuple(
                        [
                            round(non_rand_resize_scale * size)
                            for size in self._image_size
                        ]
                    )
                ),
                transforms.CenterCrop(self._image_size),
            ]
        )

    def run(self, inp: ImageClassificationInput, **kwargs) -> Dict:
        """
        Pre-Process the Inputs for DeepSparse Engine

        :param inputs: input model
        :return: list of preprocessed numpy arrays
        """

        if isinstance(inp.images, numpy.ndarray):
            image_batch = inp.images
        else:
            if isinstance(inp.images, str):
                inp.images = [inp.images]

            image_batch = list(map(self._preprocess_image, inp.images))

            # build batch
            image_batch = numpy.stack(image_batch, axis=0)

        original_dtype = image_batch.dtype
        image_batch = numpy.ascontiguousarray(image_batch, dtype=numpy.float32)

        if original_dtype == numpy.uint8:
            image_batch /= 255
            # normalize entire batch
            image_batch -= numpy.asarray(IMAGENET_RGB_MEANS).reshape((-1, 3, 1, 1))
            image_batch /= numpy.asarray(IMAGENET_RGB_STDS).reshape((-1, 3, 1, 1))

        return {"engine_inputs": [image_batch]}

    def _preprocess_image(self, image) -> numpy.ndarray:
        if isinstance(image, List):
            # image given as raw list
            image = numpy.asarray(image)
            if image.dtype == numpy.float32:
                # image is already processed, append and continue
                return image
            # assume raw image input
            # put image in PIL format for torchvision processing
            image = image.astype(numpy.uint8)
            if image.shape[0] < image.shape[-1]:
                # put channel last
                image = numpy.einsum("cwh->whc", image)
            image = Image.fromarray(image)
        elif isinstance(image, str):
            # load image from string filepath
            image = Image.open(image).convert("RGB")
        elif isinstance(image, numpy.ndarray):
            image = image.astype(numpy.uint8)
            if image.shape[0] < image.shape[-1]:
                # put channel last
                image = numpy.einsum("cwh->whc", image)
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError(
                f"inputs to {self.__class__.__name__} must be a string image "
                "file path(s), a list representing a raw image, "
                "PIL.Image.Image object(s), or a numpy array representing"
                f"the entire pre-processed batch. Found {type(image)}"
            )

        # apply resize and center crop
        image = self._pre_normalization_transforms(image)
        image_numpy = numpy.array(image)
        image.close()

        # make channel first dimension
        image_numpy = image_numpy.transpose(2, 0, 1)
        return image_numpy

    def _infer_image_size(self) -> Tuple[int, ...]:
        """
        Infer and return the expected shape of the input tensor

        :return: The expected shape of the input tensor from onnx graph
        """
        onnx_model = onnx.load(self.model_path)
        input_tensor = onnx_model.graph.input[0]
        return (
            input_tensor.type.tensor_type.shape.dim[2].dim_value,
            input_tensor.type.tensor_type.shape.dim[3].dim_value,
        )
