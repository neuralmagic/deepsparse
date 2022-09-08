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

import time
from typing import Any, Iterable, List, TextIO, Tuple, Union

import numpy

import cv2


try:
    from PIL import Image

    pil_import_error = None
except Exception as import_error:
    Image, pil_import_error = None, import_error

from pydantic import BaseModel, Field


__all__ = [
    "ComputerVisionSchema",
]


class ComputerVisionSchema(BaseModel):
    """
    A base ComputerVisionSchema to accept images, it is recommended to inherit
    ComputerVisionSchema for all Computer Vision Based tasks, this Schema provides a
    `from_files` factory method, and also specifies Field types for images
    """

    images: Union[str, List[str], List[Any], Any] = Field(
        description="List of Images to process"
    )  # List[Any] to accept List[numpy.ndarray], Any to accept numpy.ndarray

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_files(
        cls,
        files: Iterable[TextIO],
        *args,
        from_server: bool = False,
        **kwargs,
    ) -> BaseModel:
        """
        :param files: Iterable of file pointers to create ImageClassificationInput from
        :return: ImageClassificationInput constructed from files
        """
        if pil_import_error is not None:
            raise ImportError(
                "PIL is a requirement for Computer Vision pipeline schemas,"
                f" but was not found. Error:\n{pil_import_error}, "
                "try `pip install Pillow`"
            )
        images = [numpy.asarray(Image.open(file)) for file in files]
        return cls(*args, images=images, **kwargs)


def read_resize_and_transpose_u8_image(
    path: str, img_size: Tuple[int, int]
) -> numpy.ndarray:
    # NOTE: always channels last & uint8
    img = cv2.imread(path)
    img = cv2.resize(img, img_size)
    img = numpy.asarray(img)
    img = numpy.transpose(img, (2, 0, 1))  # make channels first
    return img


def make_float32(image: numpy.ndarray) -> numpy.ndarray:
    if image.dtype == numpy.float32:
        return image

    if image.dtype == numpy.uint8:
        image = image.astype(numpy.float32)
        image /= 255
        return image

    raise NotImplementedError(f"Converting {image.dtype} to float32 not implemented")


def make_uint8(image: numpy.ndarray) -> numpy.ndarray:
    if image.dtype == numpy.uint8:
        return image

    if image.dtype == numpy.float32:
        image *= 255
        image = image.astype(numpy.uint8)
        return image

    raise NotImplementedError(f"Converting {image.dtype} to uint8 not implemented")


def needs_resize(image: numpy.ndarray, img_size: Tuple[int, int]) -> bool:
    assert_3d(image)
    assert_channels_first_or_last(image)
    return (image.shape[0] == 3 and image.shape[1:] != img_size) or (
        image.shape[2] and image.shape[:2] != img_size
    )


def resize_and_channel_first(
    image: numpy.ndarray, img_size: Tuple[int, int]
) -> numpy.ndarray:
    assert_3d(image)

    # fast paths where we don't have to resize
    if image.shape[0] == 3 and image.shape[1:] == img_size:
        return image
    if image.shape[2] == 3 and image.shape[:2] == img_size:
        return numpy.transpose(image, (2, 0, 1))

    # ensure channels are actually first or last
    assert_channels_first_or_last(image)

    # slow resize cases
    if image.shape[0] == 3 and image.shape[1:] != img_size:
        image = numpy.transpose(image, (1, 2, 0))
        image = cv2.resize(image, img_size)
        image = numpy.transpose(image, (2, 0, 1))
        return numpy.ascontiguousarray(image)

    if image.shape[2] == 3 and image.shape[:2] != img_size:
        image = cv2.resize(image, img_size)
        image = numpy.transpose(image, (2, 0, 1))
        return numpy.ascontiguousarray(image)

    # should never hit here
    assert False


def assert_3d(image: numpy.ndarray):
    if image.ndim != 3:
        raise ValueError(f"Requires a 3d image, found {image.ndim} dims")


def assert_channels_first_or_last(image: numpy.ndarray):
    if not (image.shape[0] == 3 or image.shape[2] == 3):
        raise ValueError(f"Expected 3 first or last, found {image.shape}")
