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
import copy
from typing import Optional, Tuple

import numpy

import cv2
import torch
import torch.nn.functional as F
from deepsparse.yolact.schemas import YOLACTOutputSchema
from deepsparse.yolo.utils.utils import _get_color, _plot_fps


__all__ = ["annotate_image"]


def annotate_image(
    image: numpy.ndarray,
    prediction: YOLACTOutputSchema,
    images_per_sec: Optional[float] = None,
    score_threshold: float = 0.35,
) -> numpy.ndarray:
    """
    Annotate and return the img with the prediction data

    The function will:
    - draw bounding boxes on predictions of a model
    - annotate every prediction with its proper label
    - draw segmentation mask on the image

    Note: in private functions:
     - `_put_mask()`
     - `_put_annotation_text()`
     - `_put_bounding_box()`
     - `_get_text_size()`

    there are some hard-coded values that parameterize the layout of the annotations.
    You may need to adjust the parameters to improve the aesthetics of your annotations.

    :param image: original image to annotate (no pre-processing needed)
    :param prediction: predictions returned by the inference pipeline
    :param images_per_sec: optional fps value to annotate the left corner
        of the image (video) with
    :param score_threshold: minimum score a detection should have to be annotated
        on the image. Default is 0.35
    :return: the original image annotated with the given bounding boxes
        (if predictions are not None)
    """

    masks = prediction.masks[0]
    boxes = prediction.boxes[0]
    classes = prediction.classes[0]
    scores = prediction.scores[0]

    if any(x[0] is None for x in [boxes, classes, scores]):
        # no detections found
        return image

    image_res = copy.copy(image)

    masks, boxes = _resize_to_fit_img(image, masks, boxes)

    for box, mask, class_, score in zip(boxes, masks, classes, scores):
        if score > score_threshold:
            color = _get_color(class_)
            left, top, _, _ = box
            image_res = _put_mask(image=image_res, mask=mask, color=color)
            image_res = _put_bounding_box(image=image_res, box=box, color=color)

            annotation_text = f"{class_}: {score:.0%}"
            text_width, text_height = _get_text_size(annotation_text)
            image_res = _put_annotation_text(
                image=image_res,
                annotation_text=annotation_text,
                left=left,
                top=top,
                color=color,
                text_width=text_width,
                text_height=text_height,
            )

    if images_per_sec is not None:
        image_res = _plot_fps(
            img_res=image_res,
            images_per_sec=images_per_sec,
            x=20,
            y=30,
            font_scale=0.9,
            thickness=2,
        )

    return image_res


def _put_mask(
    image: numpy.ndarray, mask: torch.Tensor, color: Tuple[int, int, int]
) -> numpy.ndarray:

    img_with_mask = torch.where(
        mask[..., None].type(torch.uint8),
        torch.from_numpy(numpy.array(color)).cpu().type(torch.uint8),
        torch.from_numpy(image).cpu(),
    )
    img_with_non_transparent_masks = cv2.addWeighted(
        image, 0.3, img_with_mask.numpy(), 0.7, 0
    )
    return img_with_non_transparent_masks


def _put_annotation_text(
    image: numpy.ndarray,
    annotation_text: str,
    color: Tuple[int, int, int],
    text_width: int,
    text_height: int,
    left: int,
    top: int,
    text_font_scale: float = 0.9,
    text_thickness: int = 2,
) -> numpy.ndarray:

    image = cv2.rectangle(
        image,
        (int(left), int(top)),
        (int(left) + text_width, int(top) + text_height),
        color,
        thickness=-1,
    )

    image = cv2.putText(
        image,
        annotation_text,
        (int(left), int(top) + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_font_scale,
        (255, 255, 255),  # white text
        text_thickness,
        cv2.LINE_AA,
    )
    return image


def _put_bounding_box(
    image: numpy.ndarray,
    box: numpy.ndarray,
    color: numpy.ndarray,
    bbox_thickness: int = 2,
) -> numpy.ndarray:
    left, top, right, bottom = box
    image = cv2.rectangle(
        image,
        (int(left), int(top)),
        (int(right), int(bottom)),
        color,
        bbox_thickness,
    )
    return image


def _get_text_size(
    annotation_text: str, text_font_scale: float = 0.9, text_thickness: int = 2
) -> Tuple[int, int]:
    (text_width, text_height), text_baseline = cv2.getTextSize(
        annotation_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_font_scale,  # font scale
        text_thickness,  # thickness
    )
    text_height += text_baseline
    return text_width, text_height


def _sanitize_coordinates(
    _x1: numpy.ndarray, _x2: numpy.ndarray, img_size: int, padding: int = 0
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    This is numpy-based version of the torch.jit.script()
    `sanitize_coordinates` function.
    Used only for annotation, not the inference pipeline.

    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    Sanitizes the input coordinates so that
    x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates.
    """
    _x1 *= img_size
    _x2 *= img_size
    x1 = numpy.minimum(_x1, _x2)
    x2 = numpy.maximum(_x1, _x2)
    numpy.clip(x1 - padding, a_min=0, a_max=None, out=x1)
    numpy.clip(x2 + padding, a_min=None, a_max=img_size, out=x2)

    return x1, x2


def _resize_to_fit_img(
    original_image: numpy.ndarray, masks: numpy.ndarray, boxes: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Ported from from
    # https://github.com/neuralmagic/yolact/blob/master/layers/output_utils.py
    h, w, _ = original_image.shape

    # Resize the masks
    masks = F.interpolate(
        torch.from_numpy(masks).cpu().unsqueeze(0),
        (h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Binarize the masks
    masks.gt_(0.5)

    # Reshape the bounding boxes
    boxes = numpy.stack(boxes)

    boxes[:, 0], boxes[:, 2] = _sanitize_coordinates(boxes[:, 0], boxes[:, 2], w)
    boxes[:, 1], boxes[:, 3] = _sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)
    boxes = boxes.astype(numpy.int64)

    return masks, boxes
