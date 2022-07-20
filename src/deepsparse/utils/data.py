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

import logging
import re
from typing import List

import numpy


__all__ = [
    "arrays_to_bytes",
    "bytes_to_arrays",
    "verify_outputs",
    "parse_input_shapes",
    "numpy_softmax",
]


_LOGGER = logging.getLogger(__name__)


def arrays_to_bytes(arrays: List[numpy.array]) -> bytearray:
    """
    :param arrays: List of numpy arrays to serialize as bytes
    :return: bytearray representation of list of numpy arrays
    """
    to_return = bytearray()
    for arr in arrays:
        arr_dtype = bytearray(str(arr.dtype), "utf-8")
        arr_shape = bytearray(",".join([str(a) for a in arr.shape]), "utf-8")
        sep = bytearray("|", "utf-8")
        arr_bytes = arr.ravel().tobytes()
        to_return += arr_dtype + sep + arr_shape + sep + arr_bytes
    return to_return


def bytes_to_arrays(serialized_arr: bytearray) -> List[numpy.array]:
    """
    :param serialized_arr: bytearray representation of list of numpy arrays
    :return: List of numpy arrays decoded from input
    """
    sep = "|".encode("utf-8")
    arrays = []
    i_start = 0
    while i_start < len(serialized_arr) - 1:
        i_0 = serialized_arr.find(sep, i_start)
        i_1 = serialized_arr.find(sep, i_0 + 1)
        arr_dtype = numpy.dtype(serialized_arr[i_start:i_0].decode("utf-8"))
        arr_shape = tuple(
            [int(a) for a in serialized_arr[i_0 + 1 : i_1].decode("utf-8").split(",")]
        )
        arr_num_bytes = numpy.prod(arr_shape) * arr_dtype.itemsize
        arr_str = serialized_arr[i_1 + 1 : arr_num_bytes + (i_1 + 1)]
        arr = numpy.frombuffer(arr_str, dtype=arr_dtype).reshape(arr_shape)
        arrays.append(arr.copy())

        i_start = i_1 + arr_num_bytes + 1
    return arrays


def verify_outputs(
    outputs: List[numpy.array],
    gt_outputs: List[numpy.array],
    atol: float = 8.0e-4,
    rtol: float = 0.0,
) -> List[float]:
    """
    Compares two lists of output tensors, checking that they are sufficiently similar
    :param outputs: List of numpy arrays, usually model outputs
    :param gt_outputs: List of numpy arrays, usually reference outputs
    :param atol: Absolute tolerance for allclose
    :param rtol: Relative tolerance for allclose
    :return: The list of max differences for each pair of outputs
    """
    max_diffs = []

    if len(outputs) != len(gt_outputs):
        raise Exception(
            f"Number of outputs doesn't match, {len(outputs)} != {len(gt_outputs)}"
        )

    for i in range(len(gt_outputs)):
        gt_output = gt_outputs[i]
        output = outputs[i]

        if output.shape != gt_output.shape:
            raise Exception(
                f"Output shapes don't match, {output.shape} != {gt_output.shape}"
            )
        if type(output) != type(gt_output):
            raise Exception(
                f"Output types don't match, {type(output)} != {type(gt_output)}"
            )

        max_diff = numpy.max(numpy.abs(output - gt_output))
        max_diffs.append(max_diff)
        _LOGGER.info(
            f"Output {i}: {output.shape} {gt_output.shape} MAX DIFF: {max_diff}"
        )

        if not numpy.allclose(output, gt_output, rtol=rtol, atol=atol):
            raise Exception(
                "Output data doesn't match\n"
                f"output {i}: {output.shape} {gt_output.shape} MAX DIFF: {max_diff}\n"
                f"    mean = {numpy.mean(output):.5f} {numpy.mean(gt_output):.5f}\n"
                f"    std  = {numpy.std(output):.5f} {numpy.std(gt_output):.5f}\n"
                f"    max  = {numpy.max(output):.5f} {numpy.max(gt_output):.5f}\n"
                f"    min  = {numpy.min(output):.5f} {numpy.min(gt_output):.5f}"
            )

    return max_diffs


def parse_input_shapes(shape_string: str) -> List[List[int]]:
    """
    Reduces a string representation of a list of shapes to an actual list of shapes.
    Examples:
        "[1,2,3]" -> input0=[1,2,3]
        "[1,2,3],[4,5,6],[7,8,9]" -> input0=[1,2,3] input1=[4,5,6] input2=[7,8,9]
    """
    if not shape_string:
        return None

    shapes_list = []
    if shape_string:
        matches = re.findall(r"\[(.*?)\],?", shape_string)
        if matches:
            for match in matches:
                # Clean up stray extra brackets
                value = match.replace("[", "").replace("]", "")
                # Parse comma-separated dims into shape list
                shape = [int(s) for s in value.split(",")]
                shapes_list.append(shape)
        else:
            raise Exception(f"Can't parse input shapes parameter: {shape_string}")

    return shapes_list


def numpy_softmax(x: numpy.ndarray, axis: int = 0):
    """
    Ref: https://www.delftstack.com/howto/numpy/numpy-softmax/

    :param x: array containing values to be softmaxed
    :param axis: axis across which to perform softmax
    :return: x with values across axis softmaxed
    """
    x_max = numpy.max(x, axis=axis, keepdims=True)
    e_x = numpy.exp(x - x_max)
    e_x_sum = numpy.sum(e_x, axis=axis, keepdims=True)
    softmax_x = e_x / e_x_sum
    return softmax_x
