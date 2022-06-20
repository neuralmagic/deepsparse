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

from typing import Any, Dict, List, Tuple, Type, Union

import numpy
from pydantic import BaseModel

import cv2
from deepsparse import Pipeline
from deepsparse.yolact.schemas import YolactInputSchema, YolactOutputSchema


__all__ = ["YolactPipeline"]


@Pipeline.register(
    task="yolact",
    default_model_path=(
        "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none"
    ),
)
class YolactPipeline(Pipeline):
    """
    An inference pipeline for YOLACT, encodes the preprocessing, inference and
    postprocessing for YOLACT models into a single callable object

    TODO: Fill Out Method Implementation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        raise NotImplementedError()

    def process_inputs(
        self,
        inputs: BaseModel,
    ) -> Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the `input_schema`
            of this pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine. Can
            also include a tuple with engine inputs and special key word arguments
            to pass to process_engine_outputs to facilitate information from the raw
            inputs to postprocessing that may not be included in the engine inputs
        """
        if isinstance(inputs, list):

            if isinstance(inputs[0], str):
                raise NotImplementedError()

            elif isinstance(inputs[0], numpy.ndarray):

                def process_numpy_array(image):
                    # We start with an input image (H, W, C)
                    # and make sure it is type float
                    image = image.astype(numpy.float32)
                    # By default, when training we do not preserve
                    # aspect ratio (simply explode the image size
                    # to the default value)
                    image = cv2.resize(image, (550, 550))
                    # Apply the default backbone transform
                    image /= 255
                    image = image[:, :, [2, 0, 1]]  # BGR by default
                    return image

                return [process_numpy_array(array) for array in inputs]
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        raise NotImplementedError()

    @property
    def input_schema(self) -> Type[YolactInputSchema]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return YolactInputSchema

    @property
    def output_schema(self) -> Type[YolactOutputSchema]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return YolactOutputSchema
