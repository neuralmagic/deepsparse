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
from typing import Dict, List, Union

import numpy
import numpy as np
from pydantic import BaseModel

from constants import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from deepsparse.pipeline import Pipeline


try:
    import cv2
except ModuleNotFoundError as e:
    cv2 = None
    cv2_error = e


class ImageClassificationInput(BaseModel):
    """
    Input model for image classification
    """

    images: Union[str, numpy.ndarray, List[str]]


class ImageClassificationOutput(BaseModel):
    """
    Input model for image classification
    """

    labels: List[int]
    scores: List[float]


@Pipeline.register(task="image_classification")
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline for DeepSparse
    """

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
        :return: list of numpy arrays
        """

        # TODO: Check logic for 3-dim and 2-dim images
        images = []
        non_rand_resize_scale = 256.0 / 224.0  # standard used
        image_size = 224

        scaled_image_size = non_rand_resize_scale * image_size

        for image_file in inputs.images:
            img = cv2.imread(image_file)
            if img is not None:
                img = cv2.resize(img, (scaled_image_size, scaled_image_size))
                center = img.shape / 2
                x = center[1] - image_size / 2
                y = center[0] - image_size / 2

                crop_img = img[
                    int(y) : int(y + image_size), int(x) : int(x + image_size)
                ]

                crop_img -= np.asarray(IMAGENET_RGB_MEANS)
                crop_img /= np.asarray(IMAGENET_RGB_STDS)
                images.append(crop_img)

        return images

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
    ) -> ImageClassificationOutput:
        return ImageClassificationOutput(
            scores=numpy.max(engine_outputs[0], axis=1).tolist(),
            labels=numpy.argmax(engine_outputs[0], axis=1).tolist(),
        )

    @property
    def input_model(self) -> BaseModel:
        return ImageClassificationInput

    @property
    def output_model(self) -> BaseModel:
        return ImageClassificationOutput

    def map_labels_to_classes(
        self,
        labels: List[int],
        class_names: Union[str, Dict[int, str]],
    ) -> List[str]:
        """
        :param labels: predicted class ids
        :param class_names: A json file containing the mapping of class ids to
            class names, or a dictionary mapping class ids to class names.
        :return: Predicted class names from labels
        """

        if isinstance(class_names, str) and class_names.endswith(".json"):
            class_names = json.loads(class_names)

        predicted_class_names = [class_names[label] for label in labels]
        return predicted_class_names
