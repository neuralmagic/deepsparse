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
import numpy
import numpy as np

import cv2
import copy
import matplotlib.pyplot as plt
from deepsparse.yolact.schemas import YOLACTOutputSchema
from deepsparse.yolact.utils import sanitize_coordinates
from deepsparse.yolo.utils.utils import _get_color
from typing import Optional, Tuple


def annotate_image(
    image: numpy.ndarray,
    prediction: YOLACTOutputSchema,
    images_per_sec: Optional[float] = None,
    score_threshold: float = 0.35,
) -> numpy.ndarray:

    masks = prediction.masks[0]
    boxes = prediction.boxes[0]
    classes = prediction.classes[0]
    scores = prediction.scores[0]

    image_res = copy.copy(image)

    masks, boxes = _resize_to_fit_img(image, masks, boxes)

    for box, mask, class_, score in zip(boxes, masks, classes, scores):
        if score > score_threshold:
            colour = _get_color(class_)
            left, top, _, _ = box
            image_res = _put_mask(image = image_res, mask=mask, colour=colour)
            image_res = _put_bounding_box(image=image_res, box=box, colour=colour)

            annotation_text = f"{class_}: {score:.0%}"
            text_width, text_height = _get_text_size(annotation_text)
            image_res = _put_annotation_text(image=image_res,
                                             annotation_text = annotation_text,
                                             left = left,
                                             top = top,
                                             colour = colour, text_width = text_width, text_height=text_height
            )

    return image_res


def _put_mask(image, mask, colour):
    mask_coloured = np.where(mask[..., None], np.array(colour, dtype="uint8"), image)
    blended_image = cv2.addWeighted(image, 0.5, mask_coloured, 0.5, 0)
    return blended_image


def _put_annotation_text(
    image, annotation_text, left, top, colour, text_width, text_height
):
    image = cv2.rectangle(
        image,
        (int(left), int(top) - 33),
        (int(left) + text_width, int(top) - 28 + text_height),
        colour,
        thickness=-1,
    )

    image = cv2.putText(
        image,
        annotation_text,
        (int(left), int(top) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,  # font scale
        (255, 255, 255),  # white text
        2,  # thickness
        cv2.LINE_AA,
    )
    return image


def _put_bounding_box(image, box, colour):
    left, top, right, bottom = box
    image = cv2.rectangle(
        image,
        (int(left), int(top)),
        (int(right), int(bottom)),
        colour,
        thickness=1,
    )
    return image


def _get_text_size(annotation_text):
    (text_width, text_height), text_baseline = cv2.getTextSize(
        annotation_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,  # font scale
        2,  # thickness
    )
    text_height += text_baseline
    return text_width, text_height


def _resize_to_fit_img(original_image: numpy.ndarray, masks: numpy.ndarray, boxes) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Ported from from https://github.com/neuralmagic/yolact/blob/master/layers/output_utils.py
    h, w, _ = original_image.shape

    # TODO: Is there a faster way of interpolating? scipy?
    # Resize the masks
    masks = numpy.stack(
        [cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) for mask in masks]
    )

    # Binarize the masks
    masks = (masks > 0.5).astype(numpy.int8)

    # Reshape the bounding boxes
    boxes = numpy.stack(boxes)
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0],boxes[:, 2],w)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)
    boxes = boxes.astype(numpy.int64)

    return masks, boxes
