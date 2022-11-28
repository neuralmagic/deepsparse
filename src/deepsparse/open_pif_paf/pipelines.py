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

from typing import List, Type

import numpy
import PIL

from deepsparse.open_pif_paf.schemas import OpenPifPafInput, OpenPifPafOutput
from deepsparse.pipeline import Pipeline


__all__ = ["OpenPifPafPipeline"]


@Pipeline.register(
    task="open_pif_paf",
    default_model_path=None,
)
class OpenPifPafPipeline(Pipeline):
    """
    Open Pif Paf pipeline for DeepSparse

    :param model_path: path on local system or SparseZoo stub to load the model from
    :param engine_type: inference engine to use. Currently, supported values include
        `deepsparse` and `onnxruntime`. Default is `deepsparse`
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[OpenPifPafInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return OpenPifPafInput

    @property
    def output_schema(self) -> Type[OpenPifPafOutput]:
        """
        :return: pydantic model class that outputs to this pipeline must comply to
        """
        return OpenPifPafOutput

    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        return self.model_path

    def process_inputs(self, inputs: OpenPifPafInput) -> List[numpy.ndarray]:
        images = []
        if not isinstance(inputs.images, list):
            inputs.images = [inputs.images]

        for image in inputs.images:
            if isinstance(image, str):
                image_path = image
                with open(image_path, "rb") as f:
                    image = PIL.Image.open(f).convert("RGB")

            if image.shape[-1] == 3:
                if image.ndim == 4:
                    image = image.transpose(0, 3, 1, 2)
                else:
                    image = image.transpose(2, 0, 1)

            if image.dtype == numpy.uint8:
                image = numpy.asarray(image) / 255.0

            image = numpy.ascontiguousarray(image)
            image = image.astype(numpy.float32)
            if image.ndim == 4:
                return [image]
            else:
                images.append(image)

        return [numpy.stack(images)]

    def process_engine_outputs(
        self, fields: List[numpy.ndarray], **kwargs
    ) -> OpenPifPafOutput:
        """
        :param fields: List of two of numpy arrays of sizes:
            (B,17,5,13,17) -> CIF
            (B,19,8,13,17) -> CAF
        :return: Outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        return OpenPifPafOutput(cif=fields[0], caf=fields[1])
