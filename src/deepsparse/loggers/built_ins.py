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

from typing import Dict, List, Tuple, Union

import numpy

import torch


def image_shape(img: Union[numpy.ndarray, torch.tensor]) -> Tuple[int, int, int]:
    """
    Return the shape of the image.

    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension) tensor/array
            - the image has 3 or 1 channels
    :return: Tuple containing the image shape; three integers
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, _ = _check_valid_image(img_numpy)
    if num_dims == 4:
        img_numpy = img_numpy[0]
    return img_numpy.shape


def mean_pixels_per_channel(
    img: Union[numpy.ndarray, torch.tensor]
) -> Union[Tuple[float, float, float], Tuple[float]]:
    """
    Return the mean pixel value per image channel

    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension) tensor/array
            - the image has 3 or 1 channels
    :return: Tuple containing the mean pixel values:
        - 3 floats if image has 3 channels
        - 1 float if image has 1 channel
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, channel_dim = _check_valid_image(img_numpy)
    dims = numpy.arange(0, num_dims, 1)
    dims = numpy.delete(dims, channel_dim)
    return tuple(numpy.mean(img_numpy, axis=tuple(dims)))


def std_pixels_per_channel(
    img: Union[numpy.ndarray, torch.tensor]
) -> Union[Tuple[float, float, float], Tuple[float]]:
    """
    Return the standard deviation of pixel values per image channel
    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension) tensor/array
            - the image has 3 or 1 channels
    :return: Tuple containing the standard deviation of pixel values:
        - 3 floats if image has 3 channels
        - 1 float if image has 1 channel
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, channel_dim = _check_valid_image(img)
    dims = numpy.arange(0, num_dims, 1)
    dims = numpy.delete(dims, channel_dim)
    return tuple(numpy.std(img_numpy, axis=tuple(dims)))


def max_pixels_per_channel(
    img: Union[numpy.ndarray, torch.tensor]
) -> Union[Tuple[float, float, float], Tuple[float]]:
    """
    Return the max pixel value per image channel
    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension) tensor/array
            - the image has 3 or 1 channels
    :return: Tuple containing the max pixel values:
        - 3 floats if image has 3 channels
        - 1 float if image has 1 channel
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, channel_dim = _check_valid_image(img)
    dims = numpy.arange(0, num_dims, 1)
    dims = numpy.delete(dims, channel_dim)
    return tuple(numpy.max(img_numpy, axis=tuple(dims)))


def fraction_zeros(img: Union[numpy.ndarray, torch.tensor]) -> float:
    """
    Return the float the represents the fraction of zeros in the
    image tensor/array

    :param img: An image represented as a numpy array or a torch tensor.
       Assumptions:
           - 3 dimensional or 4 dimensional (num_batches in zeroth dimension) tensor/array
           - the image has 3 or 1 channels
    :return: A float in range from 0. to 1.
    """
    image_numpy = _assert_numpy_image(img)
    _check_valid_image(image_numpy)
    return (image_numpy.size - numpy.count_nonzero(image_numpy)) / image_numpy.size


def num_bounding_boxes(
    bboxes: Union[
        List[List[float, float, float, float]], List[List[List[float, float, float]]]
    ]
) -> int:
    """
    Extract the number of bounding boxes from the (nested) list
    of bbox corners

    :param bboxes: A (nested) list, where the leaf list has length four and contains
        float values (top left and bottom right coordinates of the bounding box corners)
    :return: Number of bounding boxes
    """
    if not bboxes:
        return 0
    # checking if
    while not (len(bboxes[0]) == 4 and isinstance(bboxes[0][0], float)):
        bboxes = bboxes[0]
        if not bboxes:
            return 0
    if not all((len(bbox) == 4 for bbox in bboxes)):
        raise ValueError(
            "Expected a single bounding box to be represented "
            "by a list containing four floating-point numbers"
        )
    return len(bboxes)


def token_count(tokens: numpy.ndarray) -> Dict[str, int]:
    count = {}
    for batch_size, token_sequence in enumerate(tokens):
        unique, counts = numpy.unique(token_sequence, return_counts=True)
        count[batch_size] = dict(zip(unique, counts))
    return count


def top_5_classes():
    pass


def _check_valid_image(img: numpy.ndarray) -> Tuple[int, int]:
    num_dims = img.ndim
    if num_dims == 4:
        img = img[0]

    channel_dim = [i for i, dim in enumerate(img.shape) if (dim == 1) or (dim == 3)]

    if img.ndim != 3:
        raise ValueError(
            "A valid image must have three or four (incl. batch dimension) dimensions"
        )

    if len(channel_dim) != 1:
        raise ValueError(
            "Could not infer a channel dimension from the image tensor/array"
        )

    channel_dim = channel_dim[0]
    return num_dims, channel_dim if num_dims == 3 else channel_dim + 1


def _assert_numpy_image(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    return img
