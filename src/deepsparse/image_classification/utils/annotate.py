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

import numpy as np

import cv2
from deepsparse.image_classification.schemas import ImageClassificationOutput


def annotate(
    img: np.ndarray,
    prediction: ImageClassificationOutput,
    target_fps: Optional[float] = None,
    display_image_shape: Tuple = (640, 640),
    x_offset: int = 25,
    y_offset: int = 25,
    font_scale: float = 0.8,
    thickness: int = 1,
) -> np.ndarray:
    """
    Annotate and return the img with the prediction data.
    Note: The layout is hard-coded, so that it supports
    `display_image_shape` = (640,640). If you wish to change
    the display image shape and preserve an aesthetic layout,
    you will need to play with `x_offset`, `y_offset`, `font_scale` and
    `thickness` arguments.

    :param img: original image to annotate
    :param prediction: predictions returned by the inference pipeline
    :param display_image_shape: target shape of the annotated image
    :param x_offset: x-coordinate of the upper,
        left corner of the prediction text box
    :param y_offset: y-coordinate of the upper,
        left corner of the prediction text box
    :param font_scale: font size of the label text
    :param thickness: thickness of the label text
    :return: the original image annotated with the prediction data
    """

    img = cv2.resize(img, display_image_shape)
    img = _put_text_box(img, y_offset, prediction)
    y_shift = copy.deepcopy(y_offset)
    for label, score in zip(prediction.labels, prediction.scores):
        # every next label annotation is placed `y_shift` pixels lower than
        # the previous one
        y_offset += y_shift
        img = _put_prediction(
            img, label, score, x_offset, y_offset, font_scale, thickness
        )
    if target_fps:
        y_offset += y_shift
        img = _put_target_fps(
            img, x_offset, y_offset, target_fps, font_scale, thickness
        )

    return img


def _put_target_fps(img, x_offset, y_offset, target_fps, font_scale, thickness):
    cv2.putText(
        img,
        f"FPS: {target_fps:.2f}",
        (int(x_offset), int(y_offset)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (0, 0, 0),  # black text
        thickness,
        cv2.LINE_AA,
    )
    return img


def _put_prediction(
    img: np.ndarray,
    label: str,
    score: str,
    x_offset: int,
    y_offset: int,
    font_scale: float,
    thickness: int,
):
    cv2.putText(
        img,
        f"{label}: {score:.2f}%",
        (int(x_offset), int(y_offset)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (0, 100, 0),  # green text
        thickness,
        cv2.LINE_AA,
    )
    return img


def _put_text_box(
    img: np.ndarray, y_offset: int, prediction: ImageClassificationOutput
):

    num_labels = len(prediction.labels)
    # text box height is a function of the
    # number of labels while
    # text box width is always half
    # the width of the image
    rect = img[0 : y_offset * (num_labels + 3), 0 : int(img.shape[1] / 2)]
    white_rect = np.ones(rect.shape, dtype=np.uint8) * 255
    rect_mixed = cv2.addWeighted(rect, 0.5, white_rect, 0.5, 1.0)

    # Putting the image back to its position
    img[0 : y_offset * (num_labels + 3), 0 : int(img.shape[1] / 2)] = rect_mixed
    return img
