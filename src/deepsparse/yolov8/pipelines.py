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

import logging
import warnings
from typing import Callable, List, Type, Union

import numpy

import torch
from deepsparse import Pipeline
from deepsparse.yolo import YOLOOutput as YOLODetOutput
from deepsparse.yolo import YOLOPipeline
from deepsparse.yolov8.schemas import YOLOSegOutput
from ultralytics.yolo.utils.ops import non_max_suppression as non_max_supression_torch
from ultralytics.yolo.utils.ops import process_mask_upsample


LOGGER = logging.getLogger(__name__)

SUPPORTED_SUBTASKS = ["detection", "segmentation"]


def non_max_supression_numpy(outputs: numpy.ndarray, **kwargs) -> torch.Tensor:
    """
    Helper function to convert engine outputs (numpy array) to torch tensor, so that
    the non_max_supression_torch function can be used.
    """
    return non_max_supression_torch(prediction=torch.from_numpy(outputs), **kwargs)


@Pipeline.register(
    task="yolov8",
    default_model_path=None,
)
class YOLOv8Pipeline(YOLOPipeline):
    def __init__(
        self,
        subtask="detection",
        nms_function: Callable = non_max_supression_numpy,
        **kwargs,
    ):
        self.subtask = subtask
        self.nms_function = nms_function
        super().__init__(nms_function=nms_function, **kwargs)

    @property
    def output_schema(self) -> Type[Union[YOLODetOutput, YOLOSegOutput]]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        if self.subtask == "detection":
            return YOLODetOutput
        elif self.subtask == "segmentation":
            return YOLOSegOutput
        else:
            raise ValueError(
                f"Specified incorrect task: {self.subtask}. "
                f"Task must be one of {SUPPORTED_SUBTASKS}"
            )

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        **kwargs,
    ) -> Union[YOLODetOutput, YOLOSegOutput]:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        if self.subtask == "segmentation":
            if len(engine_outputs) != 2:
                warnings.warn(
                    "YOLOv8 Segmentation pipeline expects 2 outputs from engine, "
                    "got {}. Assuming first output is detection output, and last "
                    "is segmentation output".format(len(engine_outputs))
                )
                engine_outputs = [engine_outputs[0], engine_outputs[5]]
            return self.process_engine_outputs_seg(
                engine_outputs=engine_outputs, **kwargs
            )

        else:
            if len(engine_outputs) != 1:
                warnings.warn(
                    "YOLOv8 Detection pipeline expects 1 output from engine, "
                    "got {}. Assuming first output is detection output".format(
                        len(engine_outputs)
                    )
                )
                engine_outputs = [engine_outputs[0]]

            return super().process_engine_outputs(
                engine_outputs=engine_outputs, **kwargs
            )

    def process_engine_outputs_seg(
        self,
        engine_outputs: List[numpy.ndarray],
        **kwargs,
    ) -> YOLOSegOutput:
        """
        The pathway for processing the outputs of the engine for YOLOv8 segmentation.
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """

        detections, mask_protos = engine_outputs

        # defined per ultralytics documentation
        num_classes = detections.shape[1] - 4

        # NMS
        detections_output = self.nms_function(
            outputs=detections,
            nc=num_classes,
            iou_thres=kwargs.get("iou_thres", 0.25),
            conf_thres=kwargs.get("conf_thres", 0.45),
            multi_label=kwargs.get("multi_label", False),
        )

        mask_protos = numpy.stack(mask_protos)
        original_image_shapes = kwargs.get("original_image_shapes")
        batch_boxes, batch_scores, batch_labels, batch_masks = [], [], [], []

        for idx, (detection_output, protos) in enumerate(
            zip(detections_output, mask_protos)
        ):
            original_image_shape = (
                original_image_shapes[idx] if idx < len(original_image_shapes) else None
            )

            bboxes = detection_output[:, :4]

            # check if empty detection
            if bboxes.shape[0] == 0:
                batch_boxes.append([None])
                batch_scores.append([None])
                batch_labels.append([None])
                batch_masks.append([None])
                continue

            bboxes = self._scale_boxes(bboxes, original_image_shape)
            scores = detection_output[:, 4]
            labels = detection_output[:, 5]
            masks_in = detection_output[:, 6:]

            protos = torch.from_numpy(protos)

            batch_boxes.append(bboxes.tolist())
            batch_scores.append(scores.tolist())
            batch_labels.append(labels.tolist())

            batch_masks.append(
                process_mask_upsample(
                    protos=protos,
                    masks_in=masks_in,
                    bboxes=bboxes,
                    shape=original_image_shape,
                ).numpy()
            )

            if self.class_names is not None:
                batch_labels_as_strings = [
                    str(int(label)) for label in batch_labels[-1]
                ]
                batch_class_names = [
                    self.class_names[label_string]
                    for label_string in batch_labels_as_strings
                ]
                batch_labels[-1] = batch_class_names

        return YOLOSegOutput(
            boxes=batch_boxes,
            scores=batch_scores,
            classes=batch_labels,
            masks=batch_masks if kwargs.get("return_masks") else None,
            intermediate_outputs=(detections, mask_protos)
            if kwargs.get("return_intermediate_outputs")
            else None,
        )
