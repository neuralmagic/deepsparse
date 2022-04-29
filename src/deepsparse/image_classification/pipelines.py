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

from deepsparse.image_classification.constants import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from deepsparse.image_classification.schemas import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from deepsparse.pipeline import Pipeline


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error


@Pipeline.register(task="image_classification")
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
    """

    def __init__(
        self,
        *,
        class_names: Union[None, str, Dict[str, str]] = None,
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

    @property
    def class_names(self) -> Optional[Dict[str, str]]:
        """
        :return: Optional dict, or json file of class names to use for
            mapping class ids to class labels
        """
        return self._class_names

    @property
    def input_model(self) -> Type[ImageClassificationInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return ImageClassificationInput

    @property
    def output_model(self) -> Type[ImageClassificationOutput]:
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
        return self.model_path

    def process_inputs(self, inputs: ImageClassificationInput) -> List[numpy.ndarray]:
        """
        Pre-Process the Inputs for DeepSparse Engine

        :param inputs: input model
        :return: list of preprocessed numpy arrays
        """

        if isinstance(inputs.images, numpy.ndarray):
            image_batch = inputs.images
        else:

            image_batch = []

            if isinstance(inputs.images, str):
                inputs.images = [inputs.images]

            for image in inputs.images:
                if cv2 is None:
                    raise RuntimeError(
                        "cv2 is required to load image inputs from file "
                        f"Unable to import: {cv2_error}"
                    )
                img = cv2.imread(image) if isinstance(image, str) else image

                img = cv2.resize(img, dsize=self._image_size)
                img = img[:, :, ::-1].transpose(2, 0, 1)
                image_batch.append(img)

            image_batch = numpy.stack(image_batch, axis=0)

        original_dtype = image_batch.dtype
        image_batch = numpy.ascontiguousarray(image_batch, dtype=numpy.float32)

        if original_dtype == numpy.uint8:

            image_batch /= 255

        # normalize entire batch
        image_batch -= numpy.asarray(IMAGENET_RGB_MEANS).reshape((-1, 3, 1, 1))
        image_batch /= numpy.asarray(IMAGENET_RGB_STDS).reshape((-1, 3, 1, 1))

        return [image_batch]

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
    ) -> ImageClassificationOutput:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_model`
            format of this pipeline
        """
        labels = numpy.argmax(engine_outputs[0], axis=1).tolist()

        if self.class_names is not None:
            labels = [self.class_names[str(class_id)] for class_id in labels]

        return ImageClassificationOutput(
            scores=numpy.max(engine_outputs[0], axis=1).tolist(),
            labels=labels,
        )

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
