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

import uuid

import numpy


__all__ = ["softmax", "generate_session_id", "pad_to_fixed_length"]


def softmax(x: numpy.ndarray) -> numpy.ndarray:
    """
    Compute softmax values for x. This function is
    against overflow/underflow by using the
    trick of shifting the input vector by subtracting
    the maximum element in it from all elements

    :param x: input array
    :return: softmax values
    """
    z = x - max(x)
    numerator = numpy.exp(z)
    denominator = numpy.sum(numerator)
    return numerator / denominator


def generate_session_id() -> str:
    """
    Generate uuid for session id. This is used to
    identify the kv cache session for the user
    """
    session_id = str(uuid.uuid4())
    return session_id


def pad_to_fixed_length(
    array: numpy.ndarray, max_len: int, axis: int = 0, value: int = 0
) -> numpy.ndarray:
    """
    Pads the array to a fixed length along the given axis.
    The padding is done on the right side of the array.

    :param array: array to pad
    :param max_len: maximum length to pad to
    :param axis: axis to pad along
    :param value: value to pad with
    :return: padded array
    """
    # per dimension padding is (before, after)
    padding = [(0, 0)] * len(array.shape)
    # for the specified axis, pad to the max length
    # (from the right side of the array)
    padding[axis] = (0, max_len - array.shape[axis])
    return numpy.pad(array, padding, mode="constant", constant_values=value)
