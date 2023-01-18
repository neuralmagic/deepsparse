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
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy


__all__ = [
    "image_shape",
    "mean_pixels_per_channel",
    "std_pixels_per_channel",
    "mean_score_per_detection",
    "std_score_per_detection",
    "fraction_zeros",
    "detected_classes",
    "number_detected_objects",
]


def image_shape(
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> Dict[str, int]:
    """
    Return the shape of the image.

    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
              tensor/array
            - the image has 3 or 1 channels
    :return: Dictionary that maps "dim_0", "dim_1" and "channels" keys to the
        appropriate integers
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, channel_dim = _check_valid_image(img_numpy)
    if num_dims == 4:
        img_numpy = img_numpy[0]
        channel_dim -= 1

    result = {"channels": img_numpy.shape[channel_dim]}
    dims_counter = 0
    for index, dim in enumerate(img_numpy.shape):
        if index != channel_dim:
            result[f"dim_{dims_counter}"] = dim
            dims_counter += 1
    return result


def mean_pixels_per_channel(
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> Dict[str, float]:
    """
    Return the mean pixel value per image channel

    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
              tensor/array
            - the image has 3 or 1 channels
    :return: Dictionary that maps channel number to the mean pixel value
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, channel_dim = _check_valid_image(img_numpy)
    dims = numpy.arange(0, num_dims, 1)
    dims = numpy.delete(dims, channel_dim)
    means = numpy.mean(img_numpy, axis=tuple(dims))
    keys = ["channel_{}".format(i) for i in range(len(means))]
    return dict(zip(keys, means))


def std_pixels_per_channel(
    img: Union[numpy.ndarray, "torch.tensor"]  # noqa F821
) -> Dict[str, float]:
    """
    Return the standard deviation of pixel values per image channel
    :param img: An image represented as a numpy array or a torch tensor.
        Assumptions:
            - 3 dimensional or 4 dimensional (num_batches in zeroth dimension)
              tensor/array
            - the image has 3 or 1 channels
    :return: Dictionary that maps channel number to the std pixel value
    """
    img_numpy = _assert_numpy_image(img)
    num_dims, channel_dim = _check_valid_image(img)
    dims = numpy.arange(0, num_dims, 1)
    dims = numpy.delete(dims, channel_dim)
    stds = tuple(numpy.std(img_numpy, axis=tuple(dims)))
    keys = ["channel_{}".format(i) for i in range(len(stds))]
    return dict(zip(keys, stds))


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


def detected_classes(
    detected_classes: List[List[Union[int, str, None]]]
) -> Dict[str, int]:
    """
    Count the number of unique classes detected in the image batch

    :param detected_classes: A nested list, that contains
        the detected classes for each sample in the batch.
        Every element of the inner list pertains to one
        set of detections in the batch. The inner list can
        either contain a single `None` value (no detection)
        or a number of integer/string representation of the
        detected classes.
     :return: Dictionary, where the keys are class labels
        and the values are their counts across the batch
    """
    counter = Counter()
    for detection in detected_classes:
        _check_valid_detection(detection)
        counter.update(detection)
    # convert keys to strings if required
    counter = {str(class_label): count for class_label, count in counter.items()}
    return counter


def number_detected_objects(
    detected_classes: List[List[Union[int, str, None]]]
) -> Dict[str, int]:
    """
    Count the number of successful detections per sample

    :param detected_classes: A nested list, that contains
        the detected classes for each sample in the batch.
        Every element of the inner list pertains to one
        set of detections in the batch. The inner list can
        either contain a single `None` value (no detection)
        or a number of integer/string representation of the
        detected classes.
    :return: Dictionary, where the keys are image indices within
        a batch and the values are the number of detected objects per image
        Example:
            {"0": 3, # 3 objects detected in the zeroth image
             "1": 0, # no objects detected in the first image
             "2": 1  # 1 object detected in the second image
             ...
             }
    """

    number_objects_per_image = {}
    for detection_idx, detection in enumerate(detected_classes):
        _check_valid_detection(detection)
        number_objects_per_image[str(detection_idx)] = (
            0 if detection == [None] else len(detection)
        )
    return number_objects_per_image


def mean_score_per_detection(
    scores: List[List[Union[None, float]]]
) -> Dict[str, float]:
    """
    Return the mean score per detection

    :param scores: A nested list, that contains
        the detected classes for each sample in the batch.
        Every element of the inner list pertains to one
        set of detections in the batch. The inner list can
        either contain a single `None` value (no detection)
        or a number of float representation of the
        score (confidence) of the detected classes.
    :return: Dictionary, where the keys are image indices within
        a batch and the values are the mean score per detection
    """
    mean_scores_per_image = {}
    for score_idx, score in enumerate(scores):
        _check_valid_score(score)
        mean_scores_per_image[str(score_idx)] = (
            0.0 if score == [None] else numpy.mean(score)
        )

    return mean_scores_per_image


def std_score_per_detection(scores: List[List[Optional[float]]]) -> Dict[str, float]:
    """
    Return the standard deviation of scores per detection

    :param scores: A nested list, that contains
        the detected classes for each sample in the batch.
        Every element of the inner list pertains to one
        set of detections in the batch. The inner list can
        either contain a single `None` value (no detection)
        or a number of float representation of the
        score (confidence) of the detected classes.
    :return: Dictionary, where the keys are image indices within
        a batch and the values are the standard deviation of scores per detection
    """
    std_scores_per_image = {}
    for score_idx, score in enumerate(scores):
        _check_valid_score(score)
        std_scores_per_image[str(score_idx)] = (
            0.0 if score == [None] else numpy.std(score)
        )

    return std_scores_per_image


def _check_valid_detection(detection: List[Union[int, str, None]]):
    if not (
        all(isinstance(det, int) for det in detection)
        or all(isinstance(det, str) for det in detection)
        or detection == [None]
    ):
        raise ValueError(
            "Detection must be either a "
            "list of integers or a list of strings or a list "
            "with a single `None` value"
        )


def _check_valid_score(score: List[Union[float, None]]):
    if not (all(isinstance(score_, float) for score_ in score) or score == [None]):
        raise ValueError(
            "Score must be either a "
            "list of floats or a list with a single `None` value"
        )


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
