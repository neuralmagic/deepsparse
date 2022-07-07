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
A  set of general helper functions for annotating the outputs
of the computer vision pipelines
"""

__all__ = [
    "put_mask",
    "put_annotation_text",
    "put_bounding_box",
    "get_text_size",
    "plot_fps",
]

from typing import Tuple

import numpy


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error


def put_mask(
    image: numpy.ndarray,
    mask: numpy.ndarray,
    color: Tuple[int, int, int],
    mask_weight: float = 0.7,
) -> numpy.ndarray:
    """

    :param image: An image (int8 values in range 0 - 255) with the shape
        (H,W,3) to apply the mask on
    :param mask: A binary mask (int8) with the shape (H, W)
    :param color: A tuple of three integers that specify the color of the mask
    :param mask_weight: A float (between 0.0 and 1.0) that specifies how
        transparent the mask is.
        The lower the number, the more "bleak" the mask is.
    :return: An original image overlayed with the mask
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Shape of an image is {image.shape}, while the shape "
            f"of a mask is {mask.shape}. The first two dimensions "
            "of an image (height and width) needs to correspond to "
            "those of the mask."
        )

    img_with_non_transparent_masks = numpy.where(
        mask[..., None], numpy.array(color, dtype="uint8"), image
    )

    img_with_non_transparent_masks = cv2.addWeighted(
        image, 1.0 - mask_weight, img_with_non_transparent_masks, mask_weight, 0
    )
    return img_with_non_transparent_masks


def put_annotation_text(
    image: numpy.ndarray,
    annotation_text: str,
    color: Tuple[int, int, int],
    left: int,
    top: int,
    text_font_scale: float = 0.9,
    text_thickness: int = 2,
) -> numpy.ndarray:
    """
    Annotates the image with the `annotation_text`.
    The text will be displayed on the image
    inside a text box
    (that fits the contents of the `annotation_text`).

    :param image: The image to be annotated
    :param annotation_text: Content to be overlayed onto the image
    :param color: The colour of the text box around the `annotation_text`
    :param left: x-coordinate of the upper left corner of the text box
    :param top: y-coordinate of the upper left corner of the text box
    :param text_font_scale: Size of the font for `annotation_text`
    :param text_thickness: Thickness of the font for `annotation_text`
    :return: `image` overlayed with the `annotation_text`
    """

    text_width, text_height = get_text_size(
        annotation_text, text_font_scale, text_thickness
    )

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


def put_bounding_box(
    image: numpy.ndarray,
    box: numpy.ndarray,
    color: numpy.ndarray,
    bbox_thickness: int = 2,
) -> numpy.ndarray:
    """
    Annotates the image with the `box` bounding box.

    :param image: The image to be annotated
    :param box: Bounding box coordinates to the put on the image
    :param color: The colour of the bounding box
    :param bbox_thickness: Thickness of the bounding box
    :return: `image` overlayed with the bounding box
    """
    left, top, right, bottom = box
    image = cv2.rectangle(
        image,
        (int(left), int(top)),
        (int(right), int(bottom)),
        color,
        bbox_thickness,
    )
    return image


def get_text_size(
    annotation_text: str, text_font_scale: float = 0.9, text_thickness: int = 2
) -> Tuple[int, int]:
    """
    Returns bounding box of the text string

    :param annotation_text: The text string
    :param text_font_scale: Size of the font for `annotation_text`
    :param text_thickness: Thickness of the font for `annotation_text`
    :return: The dimensions of the bounding boxes around the `annotation_text`
    """
    (text_width, text_height), text_baseline = cv2.getTextSize(
        annotation_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_font_scale,
        text_thickness,
    )

    return text_width, text_height


def plot_fps(
    image: numpy.ndarray,
    images_per_sec: float,
    left: int,
    top: int,
    text_font_scale: float,
    text_thickness: int,
    color=(245, 46, 6),
) -> numpy.ndarray:
    """
    Put the text box with FPS value on the image

    :param image: The image to be annotated
    :param images_per_sec: FPS value
    :param left: x-coordinate of the upper left corner of the text box
    :param top: y-coordinate of the upper left corner of the text box
    :param text_font_scale: Size of the font for FPS data
    :param text_thickness: Thickness of the font for FPS data
    :param color: The colour of the text box around the FPS data
    :return: `image` overlayed with the FPS data
    """
    annotation_text = f"FPS: {int(images_per_sec)}"
    put_annotation_text(
        image=image,
        annotation_text=annotation_text,
        color=color,
        left=left,
        top=top,
        text_font_scale=text_font_scale,
        text_thickness=text_thickness,
    )

    return image
