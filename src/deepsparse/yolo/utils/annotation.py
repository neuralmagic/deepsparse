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
)
from deepsparse.yolo.schemas import YOLOOutputSchema


def annotate_image(
    image: numpy.ndarray,
    prediction: YOLOOutputSchema,
    images_per_sec: Optional[float] = None,
    score_threshold: float = 0.35,
    model_input_size: Tuple[int, int] = None,
) -> numpy.ndarray:
    """
    Draws bounding boxes on predictions of a detection model

    :param image: original image to annotate (no pre-processing needed)
    :param prediction: predictions returned by the inference pipeline
    :param images_per_sec: optional fps value to annotate the left corner
        of the image (video) with
    :param score_threshold: minimum score a detection should have to be annotated
        on the image. Default is 0.35
    :param model_input_size: 2-tuple of expected input size for the given model to
        be used for bounding box scaling with original image. Scaling will not
        be applied if model_input_size is None. Default is None
    :return: the original image annotated with the given bounding boxes
    """
    boxes = prediction[0].boxes
    scores = prediction[0].scores
    labels = prediction[0].labels

    img_res = numpy.copy(image)

    scale_y = image.shape[0] / (1.0 * model_input_size[0]) if model_input_size else 1.0
    scale_x = image.shape[1] / (1.0 * model_input_size[1]) if model_input_size else 1.0

    for idx in range(len(boxes)):
        label = labels[idx]
        if scores[idx] > score_threshold:
            annotation_text = f"{label}: {scores[idx]:.0%}"
            color = _get_color(label)

            # bounding box points
            boxes[idx][0] *= scale_x
            boxes[idx][1] *= scale_y
            boxes[idx][2] *= scale_x
            boxes[idx][3] *= scale_y

            left, top, right, bottom = boxes[idx]

            # put text box with annotation text
            img_res = put_annotation_text(
                image=img_res,
                annotation_text=annotation_text,
                color=color,
                left=left,
                top=top,
            )

            # put bounding box
            img_res = put_bounding_box(image=img_res, box=boxes[idx], color=color)

    if images_per_sec is not None:
        img_res = plot_fps(
            img_res=img_res,
            images_per_sec=images_per_sec,
            x=20,
            y=30,
            font_scale=0.9,
            thickness=2,
        )
    return img_res


@functools.lru_cache(maxsize=None)
def _get_color(label):
    # cache color lookups
    return random.choice(COCO_CLASS_COLORS)
