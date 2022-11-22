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
import copy
from typing import Type

import numpy
import PIL
import torchvision

import cv2
import torch
from deepsparse.open_pif_paf.schemas import OpenPifPafInput, OpenPifPafOutput
from deepsparse.pipeline import Pipeline
from openpifpaf import decoder, network


__all__ = ["OpenPifPafPipeline"]


@Pipeline.register(
    task="open_pif_paf",
    default_model_path=("openpifpaf-resnet50.onnx"),
)
class OpenPifPafPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        Those two lines of code below are needed to invoke a processor.
        It is an object that translates raw network outputs into annotations.
        This is a hack to quickly instantiate a processor without having to
        write tons of boilerplate code for now.
        """
        self.model_cpu, _ = network.Factory().factory(head_metas=None)
        self.processor = decoder.factory(self.model_cpu.head_metas)

        # input image gets pre_processed before it is passed to the model
        self.pre_process_transformations = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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

        for image_path in inputs.images:
            with open(image_path, "rb") as f:
                image = PIL.Image.open(f).convert("RGB")
                image = self.pre_process_transformations(image)
                # add batch dimension and convert to numpy
                images.append(image.numpy())

        return [numpy.stack(images)], {"original_images": copy.deepcopy(inputs.images)}

    def process_engine_outputs(self, fields, **kwargs):
        """
        Fields (raw network outputs) are a list of numpy arrays of sizes:
        (B,17,5,13,17) (CIF) and (B,19,8,13,17) (CAF).
        The processor maps fields into the actual list of pose annotations
        """
        for idx, (cif, caf) in enumerate(zip(*fields)):
            annotations = self.processor._mappable_annotations(
                [torch.tensor(cif), torch.tensor(caf)], None, None
            )
            img = cv2.imread(kwargs["original_images"][idx])
            img = self._simple_plot(img, annotations)
            cv2.imwrite(f"output_{idx}.jpg", img)
        return OpenPifPafOutput(out=None)

    @staticmethod
    def _simple_plot(img, annotation):
        for keypoint in annotation:
            color = tuple(c.item() * 255 for c in torch.rand(3))
            data = keypoint.data
            for x, y, _ in data:  # last value is confidence of a keypoint
                x = int(x)
                y = int(y)
                radius = 2
                img = cv2.circle(
                    img,
                    (x, y),
                    radius,
                    color,
                    thickness=-1,
                )
        return img
