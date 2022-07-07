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
import functools
import random
from typing import Optional, Tuple


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error

import numpy

from deepsparse.vision import (
    COCO_CLASS_COLORS,
    plot_fps,
    put_annotation_text,
    put_bounding_box,
    put_mask,
)
from deepsparse.yolact.schemas import YOLACTOutputSchema


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

    Note: in functions:
     - `put_mask()`
     - `put_annotation_text()`
     - `put_bounding_box()`
     - `get_text_size()`

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

    if any(x[0] is None for x in [masks, boxes, classes, scores]):
        # no detections found
        return image

    image_res = copy.copy(image)

    masks, boxes = _resize_to_fit_img(image, masks, boxes)

    for box, mask, class_, score in zip(boxes, masks, classes, scores):
        if score > score_threshold:
            color = _get_color(class_)
            left, top, _, _ = box
            image_res = put_mask(image=image_res, mask=mask, color=color)
            image_res = put_bounding_box(image=image_res, box=box, color=color)

            annotation_text = f"{class_}: {score:.0%}"

            image_res = put_annotation_text(
                image=image_res,
                annotation_text=annotation_text,
                left=left,
                top=top,
                color=color,
            )

    if images_per_sec is not None:
        image_res = plot_fps(
            img_res=image_res,
            images_per_sec=images_per_sec,
            x=20,
            y=30,
            font_scale=0.9,
            thickness=2,
        )

    return image_res


@functools.lru_cache(maxsize=None)
def _get_color(label):
    # cache color lookups
    return random.choice(COCO_CLASS_COLORS)


def _resize_to_fit_img(
    original_image: numpy.ndarray, masks: numpy.ndarray, boxes: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Ported from from
    # https://github.com/neuralmagic/yolact/blob/master/layers/output_utils.py
    h, w, _ = original_image.shape

    # Resize the masks
    masks = numpy.stack(
        [cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) for mask in masks]
    )

    # Binarize the masks
    masks = (masks > 0.5).astype(numpy.int8)

    # Reshape the bounding boxes
    boxes = numpy.stack(boxes)
    from deepsparse.yolact.utils import sanitize_coordinates

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)
    boxes = boxes.astype(numpy.int64)

    return masks, boxes
