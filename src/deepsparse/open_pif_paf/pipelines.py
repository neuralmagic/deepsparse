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

from typing import List, Tuple, Type, Union

import numpy

import cv2
import torch
from deepsparse.open_pif_paf.schemas import OpenPifPafInput, OpenPifPafOutput
from deepsparse.pipeline import Pipeline
from deepsparse.yolact.utils import preprocess_array
from openpifpaf import decoder, network


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
     :param image_size: optional image size to override with model shape. Can
        be an int which will be the size for both dimensions, or a 2-tuple
        of the width and height sizes. Default does not modify model image shape
    """

    def __init__(
        self, *, image_size: Union[int, Tuple[int, int]] = (384, 384), **kwargs
    ):
        super().__init__(**kwargs)
        self._image_size = (
            image_size if isinstance(image_size, Tuple) else (image_size, image_size)
        )
        # necessary openpifpaf dependencies for now
        model_cpu, _ = network.Factory().factory(head_metas=None)
        self.processor = decoder.factory(model_cpu.head_metas)

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

        images = inputs.images

        if not isinstance(images, list):
            images = [images]

        image_batch = list(self.executor.map(self._preprocess_image, images))

        image_batch = numpy.concatenate(image_batch, axis=0)

        return [image_batch]

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

        data_batch, skeletons_batch, scores_batch, keypoints_batch = [], [], [], []

        for idx, (cif, caf) in enumerate(zip(*fields)):
            annotations = self.processor._mappable_annotations(
                [torch.tensor(cif), torch.tensor(caf)], None, None
            )
            data_batch.append([annotation.data.tolist() for annotation in annotations])
            skeletons_batch.append([annotation.skeleton for annotation in annotations])
            scores_batch.append([annotation.score for annotation in annotations])
            keypoints_batch.append([annotation.keypoints for annotation in annotations])

        return OpenPifPafOutput(
            data=data_batch,
            skeletons=skeletons_batch,
            scores=scores_batch,
            keypoints=keypoints_batch,
        )

    def _preprocess_image(self, image) -> numpy.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)

        return preprocess_array(image, input_image_size=self._image_size)
