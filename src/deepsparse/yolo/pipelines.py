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
import multiprocessing
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy
import onnx

from deepsparse.pipeline import Pipeline
from deepsparse.pipelines.computer_vision import (
    assert_3d,
    make_float32,
    make_uint8,
    read_resize_and_transpose_u8_image,
    resize_and_channel_first,
)
from deepsparse.utils import model_to_path
from deepsparse.yolo.schemas import YOLOInput, YOLOOutput
from deepsparse.yolo.utils import (
    COCO_CLASSES,
    YoloPostprocessor,
    get_onnx_expected_image_shape,
    modify_yolo_onnx_input_shape,
    postprocess_nms,
    yolo_onnx_has_postprocessing,
)


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error

__all__ = ["YOLOPipeline"]


@Pipeline.register(
    task="yolo",
    default_model_path=(
        "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95"
    ),
)
class YOLOPipeline(Pipeline):
    """
    Image detection YOLO pipeline for DeepSparse

    :param model_path: path on local system or SparseZoo stub to load the model from
    :param engine_type: inference engine to use. Currently, supported values
        include 'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
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
    :param image_size: optional image size to override with model shape. Can
        be an int which will be the size for both dimensions, or a 2-tuple
        of the width and height sizes. Default does not modify model image shape
    """

    def __init__(
        self,
        *,
        class_names: Optional[Union[str, Dict[str, str]]] = None,
        model_config: Optional[str] = None,
        image_size: Union[int, Tuple[int, int], None] = None,
        **kwargs,
    ):

        self._image_size = image_size
        self._onnx_temp_file = None  # placeholder for potential tmpfile reference

        super().__init__(
            **kwargs,
        )

        if isinstance(class_names, str):
            if class_names.endswith(".json"):
                class_names = json.load(open(class_names))
            elif class_names == "coco":
                class_names = COCO_CLASSES
            else:
                raise ValueError(f"Unknown class_names: {class_names}")

        if isinstance(class_names, dict):
            self._class_names = class_names
        elif isinstance(class_names, list):
            self._class_names = {
                str(index): class_name for index, class_name in enumerate(class_names)
            }
        else:
            self._class_names = None

        onnx_model = onnx.load(self.onnx_file_path)
        self.has_postprocessing = yolo_onnx_has_postprocessing(onnx_model)
        self.is_quantized = self.model_is_quantized(onnx_model=onnx_model)
        self.postprocessor = (
            None
            if self.has_postprocessing
            else YoloPostprocessor(
                image_size=self.image_size,
                cfg=model_config,
            )
        )
        self._model_config = model_config

    @property
    def model_config(self) -> str:
        return self._model_config

    @property
    def class_names(self) -> Optional[Dict[str, str]]:
        return self._class_names

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        :return: shape of image size inference is run at
        """
        return self._image_size

    @property
    def input_schema(self) -> Type[YOLOInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return YOLOInput

    @property
    def output_schema(self) -> Type[YOLOOutput]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return YOLOOutput

    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        model_path = model_to_path(self.model_path)
        if self._image_size is None:
            self._image_size = get_onnx_expected_image_shape(onnx.load(model_path))
        else:
            # override model input shape to given image size
            if isinstance(self._image_size, int):
                self._image_size = (self._image_size, self._image_size)
            self._image_size = self._image_size[:2]
            model_path, self._onnx_temp_file = modify_yolo_onnx_input_shape(
                model_path, self._image_size
            )
        return model_path

    def process_inputs(self, inputs: YOLOInput) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the `input_schema`
            of this pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """

        postprocessing_kwargs = dict(
            iou_thres=inputs.iou_thres,
            conf_thres=inputs.conf_thres,
        )

        if isinstance(inputs.images, numpy.ndarray) and inputs.images.ndim == 4:
            # NOTE: assume batch of images are already pre-preocessed
            return [inputs.images], postprocessing_kwargs

        if isinstance(inputs.images, (str, numpy.ndarray)):
            inputs.images = [inputs.images]

        images = list(self.async_map(self._preprocess_image, inputs.images))
        batch = numpy.stack(images, axis=0)
        return [batch], postprocessing_kwargs

    def _preprocess_image(self, image) -> numpy.ndarray:
        if isinstance(image, str):
            image = read_resize_and_transpose_u8_image(image, self.image_size)
            if not self.is_quantized:
                image = make_float32(image)
            return image

        if isinstance(image, numpy.ndarray):
            assert_3d(image)
            image = resize_and_channel_first(image, self.image_size)
            image = make_uint8(image) if self.is_quantized else make_float32(image)
            image = numpy.ascontiguousarray(image)
            return image

        raise ValueError(f"Expected str or numpy.ndarray, found {type(image)}")

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        **kwargs,
    ) -> YOLOOutput:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
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
        batch_output = postprocess_nms(
            batch_output,
            iou_thres=kwargs.get("iou_thres", 0.25),
            conf_thres=kwargs.get("conf_thres", 0.45),
        )

        batch_boxes, batch_scores, batch_labels = [], [], []

        for image_output in batch_output:
            batch_boxes.append(image_output[:, 0:4].tolist())
            batch_scores.append(image_output[:, 4].tolist())
            batch_labels.append(image_output[:, 5].tolist())
            if self.class_names is not None:
                batch_labels_as_strings = [
                    str(int(label)) for label in batch_labels[-1]
                ]
                batch_class_names = [
                    self.class_names[label_string]
                    for label_string in batch_labels_as_strings
                ]
                batch_labels[-1] = batch_class_names

        return YOLOOutput(
            boxes=batch_boxes,
            scores=batch_scores,
            labels=batch_labels,
        )

    def model_has_postprocessing(self, loaded_onnx_model) -> bool:
        """
        :return: True if loaded_onnx_model has postprocessing, False otherwise
        """
        # get number of dimensions in each output
        outputs_num_dims = [
            len(output.type.tensor_type.shape.dim)
            for output in loaded_onnx_model.graph.output
        ]

        # assume if only one output, then it is post-processed
        if len(outputs_num_dims) == 1:
            return True

        return all(num_dims > outputs_num_dims[0] for num_dims in outputs_num_dims[1:])

    def model_is_quantized(self, onnx_model) -> bool:
        """
        :return: True if loaded_onnx_model is quantized, False otherwise
        """
        return (
            onnx_model.graph.input[0].type.tensor_type.elem_type
            == onnx.TensorProto.UINT8
        )
