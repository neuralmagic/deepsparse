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
VitPose Pipeline
"""

from typing import Type

import numpy

import cv2
from deepsparse.pipeline import Pipeline
from deepsparse.vit_pose.schemas import VitPoseInput, VitPoseOutput
from deepsparse.yolact.utils import preprocess_array


__all__ = [
    "VitPosePipeline",
]


@Pipeline.register(
    task="vit_pose",
    default_model_path=None,
)
class VitPosePipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[VitPoseInput]:
        return VitPoseInput

    @property
    def output_schema(self) -> Type[VitPoseOutput]:
        return VitPoseOutput

    def setup_onnx_file_path(self) -> str:
        return self.model_path

    def process_inputs(self, inputs):
        images = inputs.images

        if not isinstance(images, list):
            images = [images]

        image_batch = list(self.executor.map(self._preprocess_image, images))

        image_batch = numpy.concatenate(image_batch, axis=0)

        return [image_batch], {"original_image": images}

    def process_engine_outputs(self, output, **kwargs):
        for original_image, output_ in zip(kwargs["original_image"], output):
            original_image = cv2.imread(original_image)
            heatmap = output_.sum(axis=(0, 1))
            # explode heatmap
            heatmap = cv2.resize(heatmap, (original_image.shape[:2]))
            # normalise between 0 and 255
            heatmap = (
                255 * (heatmap - numpy.min(heatmap)) / numpy.ptp(heatmap)
            ).astype(numpy.uint8)
            # apply heatmap to image
            heatmap_w_image = cv2.addWeighted(  # noqa F841
                original_image,
                0.3,
                numpy.repeat(heatmap[..., numpy.newaxis], 3, axis=2),
                0.7,
                0,
            )
        return self.output_schema(out=output)

    def _preprocess_image(self, image) -> numpy.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)

        return preprocess_array(image, input_image_size=(192, 256))
