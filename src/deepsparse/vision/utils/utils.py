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
Auxiliary function that are shared between all the vision
pipelines
"""

from typing import List, Tuple, Union

import numpy


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error

__all__ = ["preprocess_images"]


def preprocess_images(
    images: Union[str, List[str], numpy.ndarray, List[numpy.ndarray]],
    image_size: Tuple[int, int],
) -> List[numpy.ndarray]:
    """
    Takes the pipeline's raw image input and processes it, so that it can be directly
    fed into the compiled pipeline

    :param images: a list of image paths or numpy arrays that represent an image. Also
        accepts a single path/ numpy array
    :param image_size: image size expected by the model compiled by the pipeline.
    :param imagenet_preprocessing:
    :return: preprocessed numpy array (B, C, D, D); where (D,D) is image size expected
        by the network. It is a contiguous array with RGB channel order.
    """

    if not isinstance(images, list):
        images = [images]

    if isinstance(images[0], str):
        images = [cv2.imread(file_path) for file_path in images]

    # assert that images is a list of image batches (B, ., ., .)
    if images[0].ndim != 4:
        images = [numpy.expand_dims(image, axis=0) for image in images]
    # assert that images is a list of images with channels in last dim (B, ., ., C)
    images = [_assert_channels_last(image) for image in images]
    # assert that images have the expected spatial dimensions (B, D, D, C)
    if images[0].shape[1:3] != image_size:
        images = numpy.stack([cv2.resize(image, image_size) for image in images])

    return images


def preprocess_yolact(
    image: numpy.ndarray, input_image_size: Tuple[int, int]
) -> numpy.ndarray:
    """
    Preprocessing the input before feeding it into the YOLACT deepsparse pipeline

    :param image: numpy array representing input image(s). It can be batched (or not)
        and have an arbitrary dimensions order ((C,H,W) or (H,W,C)).
        It must have BGR channel order
    :param input_image_size: image size expected by the YOLACT network.
        Default is (550,550).
    :return: preprocessed numpy array (B, C, D, D); where (D,D) is image size expected
        by the network. It is a contiguous array with RGB channel order.
    """
    image = image.astype(numpy.float32)
    image = image.transpose(0, 3, 1, 2)
    image /= 255
    # BGR -> RGB
    image = image[:, (2, 1, 0), :, :]
    image = numpy.ascontiguousarray(image)

    return image


def _assert_channels_last(array: numpy.ndarray) -> numpy.ndarray:
    # make sure that the output is an array with dims
    # (B, H, W, C) or (H,W,C)
    if array.shape[1] < array.shape[2]:
        array = array.transpose(0, 2, 3, 1)
    return array
