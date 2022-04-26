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

import json
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy
import onnx

from deepsparse import Scheduler
from deepsparse.pipeline import DEEPSPARSE_ENGINE, Pipeline
from deepsparse.yolo.coco_classes import COCO_CLASSES
from deepsparse.yolo.schemas import YOLOInput, YOLOOutput
from deepsparse.yolo.utils import YoloPostprocessor, postprocess_nms


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error


@Pipeline.register(task="yolo")
class YOLOPipeline(Pipeline):
    """
    Image Segmentation YOLO pipeline for DeepSparse

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
    :param class_names: Optional string identifier, dict, or json file of
        class names to use for mapping class ids to class labels. Default is
        `coco`
    """

    def __init__(
        self,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
        class_names: Optional[Union[str, Dict[str, str]]] = "coco",
    ):
        super().__init__(
            model_path,
            engine_type,
            batch_size,
            num_cores,
            scheduler,
            input_shapes,
            alias,
        )
        self._input_shape = None

        if isinstance(class_names, str):
            if class_names.endswith(".json"):
                class_names = json.load(open(class_names))
            elif class_names == "coco":
                class_names = COCO_CLASSES
            else:
                raise ValueError(f"Unknown class_names: {class_names}")

        if isinstance(class_names, dict):
            self.class_names = class_names
        elif isinstance(class_names, list):
            self.class_names = {
                str(index): class_name for index, class_name in enumerate(class_names)
            }
        else:
            raise ValueError(
                "class_names must be a str identifier, dict, json file, or "
                f"list of class names got {type(class_names)}"
            )

        self.onnx_model = onnx.load(self.engine.model_path)

        self.postprocessor = (
            None
            if self.has_postprocessing
            else YoloPostprocessor(image_size=self.input_shape)
        )

    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        return self.model_path

    def process_inputs(self, inputs: YOLOInput) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the `input_model`
            of this pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        image_batch = []

        if isinstance(inputs.images, str):
            inputs.images = [inputs.images]

        for image in inputs.images:
            img = cv2.imread(image) if isinstance(image, str) else image

            img = cv2.resize(img, dsize=self.input_shape)
            img = img[:, :, ::-1].transpose(2, 0, 1)

            image_batch.append(img)

        image_batch = numpy.stack(image_batch, axis=0)
        image_batch = numpy.ascontiguousarray(
            image_batch, dtype=numpy.int8 if self.is_quantized else numpy.float32
        )
        image_batch /= 255

        return [image_batch]

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
    ) -> YOLOOutput:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_model`
            format of this pipeline
        """

        # post-processing
        if self.postprocessor:
            batch_output = self.postprocessor.pre_nms_postprocess(engine_outputs)
        else:
            batch_output = engine_outputs[
                0
            ]  # post-processed values stored in first output

        # NMS
        batch_output = postprocess_nms(batch_output)

        batch_predictions, batch_boxes, batch_scores, batch_labels = [], [], [], []

        for image_output in batch_output:
            batch_predictions.append(image_output.tolist())
            batch_boxes.append(image_output[:, 0:4].tolist())
            batch_scores.append(image_output[:, 4].tolist())
            batch_labels.append(
                [
                    self.class_names[str(class_ids)]
                    for class_ids in image_output[:, 5].astype(int)
                ]
            )

        return YOLOOutput(
            predictions=batch_predictions,
            boxes=batch_boxes,
            scores=batch_scores,
            labels=batch_labels,
        )

    @property
    def input_model(self) -> Type[YOLOInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return YOLOInput

    @property
    def output_model(self) -> Type[YOLOOutput]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return YOLOOutput

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Returns the expected shape of the input tensor

        :return: The expected shape of the input tensor from onnx graph
        """
        if self._input_shape is None:
            self._input_shape = self._infer_image_shape()
        return self._input_shape

    def _infer_image_shape(self) -> Tuple[int, ...]:
        """
        Infer and return the expected shape of the input tensor

        :return: The expected shape of the input tensor from onnx graph
        """
        input_tensor = self.onnx_model.graph.input[0]
        return (
            input_tensor.type.tensor_type.shape.dim[2].dim_value,
            input_tensor.type.tensor_type.shape.dim[3].dim_value,
        )

    @property
    def has_postprocessing(self) -> bool:
        """
        :return: True if onnx_model has postprocessing, False otherwise
        """
        # get number of dimensions in each output
        outputs_num_dims = [
            len(output.type.tensor_type.shape.dim)
            for output in self.onnx_model.graph.output
        ]

        # assume if only one output, then it is post-processed
        if len(outputs_num_dims) == 1:
            return True

        return all(num_dims > outputs_num_dims[0] for num_dims in outputs_num_dims[1:])

    @property
    def is_quantized(self) -> bool:
        """
        :return: True if onnx_model is quantized, False otherwise
        """
        return (
            self.onnx_model.graph.input[0].type.tensor_type.elem_type
            == onnx.TensorProto.INT8
        )
