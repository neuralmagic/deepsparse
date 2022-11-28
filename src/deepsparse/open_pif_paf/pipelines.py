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
OpenPifPafPipeline
"""
from typing import Type

import numpy
import PIL

from deepsparse.open_pif_paf.schemas import OpenPifPafInput, OpenPifPafOutput
from deepsparse.pipeline import Pipeline


__all__ = ["OpenPifPafPipeline"]


@Pipeline.register(
    task="open_pif_paf",
    default_model_path="openpifpaf-resnet50.onnx",
)
class OpenPifPafPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[OpenPifPafInput]:
        return OpenPifPafInput

    @property
    def output_schema(self) -> Type[OpenPifPafOutput]:
        return OpenPifPafOutput

    def setup_onnx_file_path(self) -> str:
        return self.model_path

    def process_inputs(self, inputs):
        images = []

        for image in inputs.images:
            if isinstance(image, str):
                image_path = image
                with open(image_path, "rb") as f:
                    image = PIL.Image.open(f).convert("RGB")
            image = numpy.asarray(image) / 255.0
            # maybe we should use the same preprocessing as in the original repo
            # but does not seem to make a difference
            image = (
                image.astype(numpy.float32).transpose(2, 0, 1)
                if image.shape[-1] == 3
                else image.astype(numpy.float32)
            )
            image = numpy.ascontiguousarray(image)
            images.append(image)

        return [numpy.stack(images)]

    def process_engine_outputs(self, fields, **kwargs):
        """
        Fields (raw network outputs) are a list of numpy arrays of sizes:
        (B,17,5,13,17) (CIF) and (B,19,8,13,17) (CAF).
        The processor maps fields into the actual list of pose annotations
        """
        return OpenPifPafOutput(cif=fields[0], caf=fields[1])
