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
import copy
from typing import Type

import numpy
import PIL

import cv2
from deepsparse.pipeline import Pipeline
from deepsparse.vit_pose.schemas import VitPoseInput, VitPoseOutput


__all__ = [
    "VitPosePipeline",
]


@Pipeline.register(
    task="vit_pose",
    default_model_path=("vit_pose.onnx"),
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
        images = []

        for image_path in inputs.images:
            with open(image_path, "rb") as f:
                image = PIL.Image.open(f).convert("RGB")
                image = numpy.asarray(image) / 255.0
                image = cv2.resize(image, (192, 256))
                image = image.astype(numpy.float32).transpose(2, 0, 1)
                image = numpy.ascontiguousarray(image)
                images.append(image)

        return [numpy.stack(images)], {"original_images": copy.deepcopy(inputs.images)}

    def process_engine_outputs(self, out, **kwargs):
        return self.output_schema(out=out)
