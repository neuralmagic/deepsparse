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
"""
Helpers and Utilities for
Image Classification annotation script
"""
from typing import Tuple

import numpy as numpy

import cv2
from deepsparse.image_classification.schemas import ImageClassificationOutput


__all__ = ["annotate_image"]


def annotate_image(
    image: numpy.ndarray,
    prediction: ImageClassificationOutput,
    images_per_sec: float,
    x_offset: int = 10,
    y_offset: int = 15,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> numpy.ndarray:
    """
    Annotate and return the img with the prediction data.
    Predictions are written into lower-left corner of the image.
    Note: The layout by default is hard-coded to support
    (640,360) shape. If you wish to change
    the display image shape and preserve an aesthetic layout,
    you will need to play with `x_offset`, `y_offset`, `font_scale` and
    `thickness` arguments.

    :param image: original image to annotate
    :param prediction: predictions returned by the inference pipeline
    :param images_per_sec: optional FPS value to display on the frame
    :param x_offset: x-coordinate of the upper,
        left corner of the prediction text box
    :param y_offset: y-coordinate of the upper,
        left corner of the prediction text box
    :param font_scale: font size of the label text
    :param thickness: thickness of the label text
    :return: the original image annotated with the prediction data
    """
    is_video = images_per_sec is not None

    image = _put_text_box(image, y_offset, prediction, is_video)
    y_shift = image.shape[0]
    for label, score in zip(reversed(prediction.labels), reversed(prediction.scores)):
        # every next label annotation is placed `y_shift` pixels higher than
        # the previous one
        y_shift -= y_offset
        image = _put_text(
            image, f"{label}: {score:.2f}%", x_offset, y_shift, font_scale, thickness
        )
    if is_video:
        # for the video annotation, additionally
        # include the FPS information
        y_shift -= 2 * y_offset
        image = _put_text(
            image,
            f"FPS: {int(images_per_sec)}",
            x_offset,
            y_shift,
            font_scale,
            thickness,
            (245, 46, 6),
        )

    return image


def _put_text(
    img: numpy.ndarray,
    text: str,
    x_offset: int,
    y_offset: int,
    font_scale: float,
    thickness: int,
    text_colour: Tuple = (0, 0, 0),
):
    cv2.putText(
        img,
        text,
        (int(x_offset), int(y_offset)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        text_colour,
        thickness,
        cv2.LINE_AA,
    )
    return img


def _put_text_box(
    img: numpy.ndarray,
    y_offset: int,
    prediction: ImageClassificationOutput,
    is_video: bool,
):

    num_text_rows = (
        len(prediction.labels) + 2
    )  # no of text rows + 2 margins (top, bottom)
    num_text_rows += 2 * int(is_video)  # add two more text rows for FPS data
    # text box height is a function of the
    # number of labels while
    # text box width is always half
    # the width of the image
    rect = img[-y_offset * num_text_rows : img.shape[0], 0 : int(img.shape[1] / 2)]
    white_rect = numpy.ones(rect.shape, dtype=numpy.uint8) * 255
    rect_mixed = cv2.addWeighted(rect, 0.5, white_rect, 0.5, 1.0)

    # Putting the image back to its position
    img[
        -y_offset * num_text_rows : img.shape[0], 0 : int(img.shape[1] / 2)
    ] = rect_mixed
    return img
