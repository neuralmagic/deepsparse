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

import torch
from deepsparse import Pipeline
from deepsparse.utils import model_to_path
from deepsparse.yolact.schemas import YOLACTInputSchema, YOLACTOutputSchema
from deepsparse.yolact.utils import decode, detect, postprocess, preprocess_array
from deepsparse.yolo.utils import COCO_CLASSES


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error

__all__ = ["YOLACTPipeline"]


@Pipeline.register(
    task="yolact",
    default_model_path=(
        "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none"
    ),
)
class YOLACTPipeline(Pipeline):
    """
    Image classification pipeline for DeepSparse

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
    :param image_size: optional image size to override with model shape. Can
        be an int which will be the size for both dimensions, or a 2-tuple
        of the width and height sizes. Default does not modify model image shape
    :param top_k: The integer that specifies how many most probable classes
        we want to fetch per image. Default is 50.
    """

    def __init__(
        self,
        *,
        class_names: Optional[Union[str, Dict[str, str], List[str]]] = None,
        image_size: Union[int, Tuple[int, int]] = (550, 550),
        top_k: int = 50,
        **kwargs,
    ):

        self._image_size = (
            image_size if isinstance(image_size, Tuple) else (image_size, image_size)
        )
        self.top_k = top_k

        super().__init__(**kwargs)

        if isinstance(class_names, str):
            if class_names.endswith(".json"):
                class_names = json.load(open(class_names))
            elif class_names.lower() == "coco":
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

    @property
    def class_names(self) -> Optional[Dict[str, str]]:
        return self._class_names

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        :return: shape of image size inference is run at
        """
        return self._image_size

    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        return model_to_path(self.model_path)

    def process_inputs(
        self,
        inputs: YOLACTInputSchema,
    ) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the `input_schema`
            of this pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine. Can
            also include a tuple with engine inputs and special key word arguments
            to pass to process_engine_outputs to facilitate information from the raw
            inputs to postprocessing that may not be included in the engine inputs
        """
        images = inputs.images

        if not isinstance(images, list):
            images = [images]

        image_batch = list(self.executor.map(self._preprocess_image, inputs.images))

        image_batch = numpy.concatenate(image_batch, axis=0)

        postprocessing_kwargs = dict(
            confidence_threshold=inputs.confidence_threshold,
            nms_threshold=inputs.nms_threshold,
            score_threshold=inputs.score_threshold,
            top_k_preprocessing=inputs.top_k_preprocessing,
            max_num_detections=inputs.max_num_detections,
            return_masks=inputs.return_masks,
        )
        return [image_batch], postprocessing_kwargs

    @staticmethod
    def join_engine_outputs(
        batch_outputs: List[List[numpy.ndarray]],
    ) -> List[numpy.ndarray]:
        boxes, confidence, masks, priors, protos = Pipeline.join_engine_outputs(
            batch_outputs
        )

        # priors never has a batch dimension
        # so the above step doesn't concat along a batch dimension
        # reshape into a batch dimension
        num_priors = boxes.shape[1]
        batch_priors = numpy.reshape(priors, (-1, num_priors, 4))

        # all the priors should be equal, so only use the first one
        assert (batch_priors == batch_priors[0]).all()
        return [boxes, confidence, masks, batch_priors[0], protos]

    def _preprocess_image(self, image) -> numpy.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)

        return preprocess_array(image)

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> YOLACTOutputSchema:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        boxes, confidence, masks, priors, protos = engine_outputs

        boxes = torch.from_numpy(boxes).cpu()
        confidence = torch.from_numpy(confidence).cpu()
        masks = torch.from_numpy(masks).cpu()
        priors = torch.from_numpy(priors).cpu()
        protos = torch.from_numpy(protos).cpu()

        batch_size, num_priors, _ = boxes.size()

        # Preprocess every image in the batch individually
        batch_classes, batch_scores, batch_boxes, batch_masks = [], [], [], []

        for batch_idx, (
            boxes_single_image,
            masks_single_image,
            confidence_single_image,
        ) in enumerate(zip(boxes, masks, confidence)):

            decoded_boxes = decode(boxes_single_image, priors)

            results = detect(
                confidence_single_image,
                decoded_boxes,
                masks_single_image,
                confidence_threshold=kwargs["confidence_threshold"],
                nms_threshold=kwargs["nms_threshold"],
                max_num_detections=kwargs["max_num_detections"],
                top_k=kwargs["top_k_preprocessing"],
            )
            if results is not None and protos is not None:
                results["protos"] = protos[batch_idx]

            if results:
                classes, scores, boxes, masks = postprocess(
                    dets=results,
                    crop_masks=True,
                    score_threshold=kwargs["score_threshold"],
                )

                classes = classes.numpy()
                scores = scores.numpy()
                boxes = boxes.numpy()
                masks = masks.numpy()

                # Choose the best k detections (taking into account all the classes)
                idx = numpy.argsort(scores)[::-1][: self.top_k]

                batch_classes.append(
                    list(map(self.class_names.__getitem__, map(str, classes[idx])))
                    if self.class_names is not None
                    else classes[idx].tolist()
                )
                batch_scores.append(scores[idx].tolist())
                batch_boxes.append(boxes[idx].tolist())
                batch_masks.append(masks[idx])

            else:
                batch_classes.append([None])
                batch_scores.append([None])
                batch_boxes.append([None])
                batch_masks.append(None)

        return YOLACTOutputSchema(
            classes=batch_classes,
            scores=batch_scores,
            boxes=batch_boxes,
            masks=batch_masks if kwargs.get("return_masks") else None,
        )

    @property
    def input_schema(self) -> Type[YOLACTInputSchema]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return YOLACTInputSchema

    @property
    def output_schema(self) -> Type[YOLACTOutputSchema]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return YOLACTOutputSchema
