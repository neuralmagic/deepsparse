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
from typing import List, Tuple, Union

import numpy


__all__ = [
    "arrays_to_bytes",
    "bytes_to_arrays",
    "verify_outputs",
    "parse_input_shapes",
    "numpy_softmax",
    "split_engine_inputs",
    "join_engine_outputs",
    "prep_for_serialization",
]

from pydantic import BaseModel


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
        if type(output) is not type(gt_output):
            raise Exception(
                f"Output types don't match, {type(output)} != {type(gt_output)}"
            )

        max_diff = numpy.max(numpy.abs(output - gt_output))
        max_diffs.append(max_diff)
        _LOGGER.info(
            f"Output {i}: {output.shape} {gt_output.shape} MAX DIFF: {max_diff}"
        )

        if not numpy.allclose(output, gt_output, rtol=rtol, atol=atol):
            _LOGGER.error(
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


def split_engine_inputs(
    items: List[numpy.ndarray], batch_size: int
) -> Tuple[List[List[numpy.ndarray]], int]:
    """
    Splits each item into numpy arrays with the first dimension == `batch_size`.

    For example, if `items` has three numpy arrays with the following
    shapes: `[(4, 32, 32), (4, 64, 64), (4, 128, 128)]`

    Then with `batch_size==4` the output would be:
    ```
    [[(4, 32, 32), (4, 64, 64), (4, 128, 128)]]
    ```

    Then with `batch_size==2` the output would be:
    ```
    [
        [(2, 32, 32), (2, 64, 64), (2, 128, 128)],
        [(2, 32, 32), (2, 64, 64), (2, 128, 128)],
    ]
    ```

    Then with `batch_size==1` the output would be:
    ```
    [
        [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
        [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
        [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
        [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
    ]
    ```

    In the case where the total input batch size isn't divisble by `batch_size`, it
    will pad the last mini batch. Look at `padding_is_needed`

    :param items: list of numpy arrays to split
    :param batch_size: size of each batch to split into

    :return: list of batches, where each batch is a list of numpy arrays,
        as well as the total batch size
    """
    # The engine expects to recieve data in numpy format, so at this point it should be
    assert all(isinstance(item, numpy.ndarray) for item in items)

    # Check that all inputs have the same batch size
    total_batch_size = items[0].shape[0]
    if not all(arr.shape[0] == total_batch_size for arr in items):
        raise ValueError("Not all inputs have matching batch size")

    batches = []
    for section_idx in range(0, total_batch_size, batch_size):
        padding_is_needed = section_idx + batch_size > total_batch_size
        if padding_is_needed:
            # If we can't evenly divide with batch size, pad the last batch
            input_sections = []
            for arr in items:
                pads = ((0, section_idx + batch_size - total_batch_size),) + (
                    (0, 0),
                ) * (arr.ndim - 1)
                section = numpy.pad(
                    arr[section_idx : section_idx + batch_size], pads, mode="edge"
                )
                input_sections.append(section)
            batches.append(input_sections)
        else:
            # Otherwise we just take our slice as the batch
            batches.append(
                [arr[section_idx : section_idx + batch_size] for arr in items]
            )

    return batches, total_batch_size


def join_engine_outputs(
    batch_outputs: List[List[numpy.ndarray]], orig_batch_size: int
) -> List[numpy.ndarray]:
    """
    Joins list of engine outputs together into one list using `numpy.stack`.
    If the batch size doesn't evenly divide into the available batches, it will cut off
    the remainder as padding.

    This is the opposite of `split_engine_inputs` and is meant to be used in tandem.

    :param batch_outputs: List of engine outputs
    :param orig_batch_size: The original batch size of the inputs
    :return: List of engine outputs joined together
    """
    assert all(isinstance(item, (List, Tuple)) for item in batch_outputs)

    candidate_output = list(map(numpy.concatenate, zip(*batch_outputs)))

    # If we can't evenly divide with batch size, remove the remainder as padding
    if candidate_output[0].shape[0] > orig_batch_size:
        for i in range(len(candidate_output)):
            candidate_output[i] = candidate_output[i][:orig_batch_size]

    return candidate_output


def prep_for_serialization(
    data: Union[BaseModel, numpy.ndarray, list]
) -> Union[BaseModel, list]:
    """
    Prepares input data for JSON serialization by converting any numpy array
    field to a list. For large numpy arrays, this operation will take a while to run.

    :param data: data to that is to be processed before
        serialization. Nested objects are supported.
    :return: Pipeline_outputs with potential numpy arrays
        converted to lists
    """
    if isinstance(data, BaseModel):
        for field_name in data.__fields__.keys():
            field_value = getattr(data, field_name)
            if isinstance(field_value, (numpy.ndarray, BaseModel, list)):
                setattr(
                    data,
                    field_name,
                    prep_for_serialization(field_value),
                )

    elif isinstance(data, numpy.ndarray):
        data = data.tolist()

    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = prep_for_serialization(value)

    elif isinstance(data, dict):
        for key, value in data.items():
            data[key] = prep_for_serialization(value)

    return data
