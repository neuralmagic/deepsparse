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

"""
Image classification pipeline
"""
import json
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy
import onnx
from PIL import Image
from torchvision import transforms

from deepsparse.image_classification.schemas import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from deepsparse.pipeline import Pipeline
from deepsparse.utils import model_to_path
from deepsparse.vision import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS, preprocess_images


__all__ = [
    "ImageClassificationPipeline",
]


@Pipeline.register(
    task="image_classification",
    default_model_path=(
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
        "imagenet/pruned85_quant-none-vnni"
    ),
)
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline for DeepSparse

    :param model_path: path on local system or SparseZoo stub to load the model from
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: static batch size to use for inference. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param class_names: Optional dict, or json file of class names to use for
        mapping class ids to class labels. Default is None
    :param top_k: The integer that specifies how many most probable classes
        we want to fetch per image. Default is 1.
    """

    def __init__(
        self,
        *,
        class_names: Union[None, str, Dict[str, str]] = None,
        top_k: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(class_names, str) and class_names.endswith(".json"):
            self._class_names = json.load(open(class_names))
        elif isinstance(class_names, dict):
            self._class_names = class_names
        else:
            self._class_names = None

        self._image_size = self._infer_image_size()
        self.top_k = top_k

        # torchvision transforms for raw inputs
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

    @property
    def class_names(self) -> Optional[Dict[str, str]]:
        """
        :return: Optional dict, or json file of class names to use for
            mapping class ids to class labels
        """
        return self._class_names

    @property
    def input_schema(self) -> Type[ImageClassificationInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return ImageClassificationInput

    @property
    def output_schema(self) -> Type[ImageClassificationOutput]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return ImageClassificationOutput

    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """

        return model_to_path(self.model_path)

    def process_inputs(self, inputs: ImageClassificationInput) -> List[numpy.ndarray]:
        """
        Pre-Process the Inputs for DeepSparse Engine

        :param inputs: input model
        :return: list of preprocessed numpy arrays
        """
        # `preprocess_images` returns a list of image_batches
        # with dimensions (B, D, D, C)
        images_input = preprocess_images(
            images=inputs.images, image_size=self._image_size
        )

        images_input_numpy = numpy.array(images_input)
        N, B, D, _, C = images_input_numpy.shape
        # apply resize and center crop
        images_input_pil = [
            self._pre_normalization_transforms(Image.fromarray(image))
            for image in images_input_numpy.reshape(-1, D, D, C)
        ]
        images_input = numpy.array([numpy.array(image) for image in images_input_pil])
        images_input = [image for image in images_input.reshape(N, B, D, D, C)]
        # make channel first dimension
        images_input = [image.transpose(0, 3, 1, 2) for image in images_input]
        # if pixel values not in range 0.0 - 1.0, make them so
        if not images_input[0].max() <= 1.0:
            images_input = [image / 255 for image in images_input]

        # apply ImageNet specific preprocessing
        for idx, image in enumerate(images_input):
            image -= numpy.asarray(IMAGENET_RGB_MEANS).reshape((-1, 3, 1, 1))
            image /= numpy.asarray(IMAGENET_RGB_STDS).reshape((-1, 3, 1, 1))
            image = numpy.ascontiguousarray(image, dtype=numpy.float32)
            images_input[idx] = image

        return images_input

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
    ) -> ImageClassificationOutput:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        labels, scores = [], []
        for prediction_batch in engine_outputs[0]:
            label = (-prediction_batch).argsort()[: self.top_k]
            score = prediction_batch[label]
            labels.append(label)
            scores.append(score.tolist())

        if self.class_names is not None:
            labels = numpy.vectorize(self.class_names.__getitem__)(labels)

        if isinstance(labels, numpy.ndarray):
            labels = labels.tolist()

        if len(labels) == 1:
            labels = labels[0]
            scores = scores[0]

        return self.output_schema(
            scores=scores,
            labels=labels,
        )

    def _infer_image_size(self) -> Tuple[int, ...]:
        """
        Infer and return the expected shape of the input tensor

        :return: The expected shape of the input tensor from onnx graph
        """
        onnx_model = onnx.load(self.onnx_file_path)
        input_tensor = onnx_model.graph.input[0]
        return (
            input_tensor.type.tensor_type.shape.dim[2].dim_value,
            input_tensor.type.tensor_type.shape.dim[3].dim_value,
        )
