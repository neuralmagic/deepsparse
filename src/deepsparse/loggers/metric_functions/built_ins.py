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
The set of all the built-in metric functions
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy


__all__ = [
    "identity",
    "image_shape",
    "mean_pixels_per_channel",
    "std_pixels_per_channel",
    "max_pixels_per_channel",
    "fraction_zeros",
    "bounding_box_count",
]


def identity(x: Any):
    """
    Simple identity function

    :param x: Any object
    :return: The same object
    """
    return x


def image_shape(
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> Tuple[int, int, int]:
    """
    Return the shape of the image.

    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
              tensor/array
            - the image has 3 or 1 channels
    :return: Tuple containing the image shape; three integers
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, _ = _check_valid_image(img_numpy)
    if num_dims == 4:
        img_numpy = img_numpy[0]
    return img_numpy.shape


def mean_pixels_per_channel(
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> Union[Tuple[float, float, float], Tuple[float]]:
    """
    Return the mean pixel value per image channel

    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
              tensor/array
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
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> Union[Tuple[float, float, float], Tuple[float]]:
    """
    Return the standard deviation of pixel values per image channel
    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
              tensor/array
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
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> Union[Tuple[float, float, float], Tuple[float]]:
    """
    Return the max pixel value per image channel
    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
              tensor/array
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


def fraction_zeros(img: Union[numpy.ndarray, "torch.tensor"]) -> float:  # noqa F821
    """
    Return the float the represents the fraction of zeros in the
    image tensor/array

    :param img: An image represented as a numpy array or a torch tensor.
       Assumptions:
           - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
             tensor/array
           - the image has 3 or 1 channels
    :return: A float in range from 0. to 1.
    """
    image_numpy = _assert_numpy_image(img)
    _check_valid_image(image_numpy)
    return (image_numpy.size - numpy.count_nonzero(image_numpy)) / image_numpy.size


def bounding_box_count(bboxes: List[List[Optional[List[float]]]]) -> Dict[int, int]:
    """
    Extract the number of bounding boxes from the (nested) list of bbox corners

    :param bboxes: A (nested) list, where the leaf list has length four and contains
        float values (top left and bottom right coordinates of the bounding box corners)
    :return: Dictionary, where the keys are image indices within
        a batch and the values are the bbox counts
    """
    if not bboxes or _is_nested_list_empty(bboxes):
        return 0

    if not (isinstance(bboxes[0][0][0], float) and len(bboxes[0][0]) == 4):
        raise ValueError(
            "A valid argument `bboxes` should be of "
            "type: List[List[Optional[List[float]]]])."
        )

    bboxes_count = {}
    for batch_idx, bboxes_ in enumerate(bboxes):
        num_bboxes = len(bboxes_)
        bboxes_count[batch_idx] = num_bboxes

    return bboxes_count


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


def _assert_numpy_image(
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> numpy.ndarray:
    if hasattr(img, "numpy"):
        img = img.numpy()
    return img


def _is_nested_list_empty(nested_list: List) -> bool:
    if not nested_list:
        return True
    if isinstance(nested_list[0], list):
        return _is_nested_list_empty(nested_list[0])
    return False
