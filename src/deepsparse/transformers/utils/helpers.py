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
import uuid
from typing import List, Optional, Tuple

import numpy
import onnx

from deepsparse.utils.onnx import (
    CACHE_INPUT_NAME,
    default_cached_outputs,
    translate_onnx_type_to_numpy,
)
from sparsezoo.utils import save_onnx


__all__ = [
    "overwrite_onnx_model_inputs_for_kv_cache_models",
    "generate_session_id",
    "pad_to_fixed_length",
    "softmax",
]

_LOGGER = logging.getLogger(__name__)


def overwrite_onnx_model_inputs_for_kv_cache_models(
    onnx_file_path: str,
    sequence_length: int = 128,
    input_ids_length: int = 1,
    batch_size: int = 1,
) -> Tuple[str, List[int], Optional[numpy.dtype]]:
    """
    Enforces the appropriate input shapes for the onnx model, as well as
    checks whether kv cache is enabled or not.

    :param onnx_file_path: The path to the onnx model file that will be
        overwritten with the new input shapes
    :param batch_size: The batch size to use for the input
    :param sequence_length: The sequence length to use for the input
    :param input_ids_length: The length of input_ids
    :return: A tuple that contains:
        -   the path to the onnx model file that has been overwritten
            with the new input shapes
        -   boolean list, where elements are set to True if the
            corresponding model output should be cached or False
            if not.
        -  the data type of the kv cache. If the model does not
            use kv cache, then the data type is None
    """
    model = onnx.load(onnx_file_path, load_external_data=False)
    initializer_input_names = set(node.name for node in model.graph.initializer)
    external_inputs = [
        inp for inp in model.graph.input if inp.name not in initializer_input_names
    ]
    for external_input in external_inputs:
        # overwrite the batch size for all the inputs
        external_input.type.tensor_type.shape.dim[0].dim_value = batch_size

        if external_input.name in ["input_ids", "positions"]:
            external_input.type.tensor_type.shape.dim[1].dim_value = input_ids_length
        elif external_input.name == "attention_mask":
            external_input.type.tensor_type.shape.dim[1].dim_value = sequence_length
        elif external_input.name.startswith(CACHE_INPUT_NAME):
            external_input.type.tensor_type.shape.dim[2].dim_value = (
                sequence_length - input_ids_length
            )
        else:
            raise ValueError(f"Unexpected external input name: {external_input.name}")

    _LOGGER.info(
        "Overwriting in-place the input shapes "
        f"of the transformer model at {onnx_file_path}"
    )
    save_onnx(model, onnx_file_path)

    output_indices_to_be_cached = default_cached_outputs(model)

    kv_cache_data_type = None
    if sum(output_indices_to_be_cached):
        kv_cache_elem_type = next(
            inp for inp in model.graph.input if inp.name.startswith(CACHE_INPUT_NAME)
        ).type.tensor_type.elem_type
        kv_cache_data_type = translate_onnx_type_to_numpy(kv_cache_elem_type)

    return onnx_file_path, output_indices_to_be_cached, kv_cache_data_type


def generate_session_id() -> str:
    """
    Generate uuid for session id. This is used to
    identify the kv cache session for the user
    """
    session_id = str(uuid.uuid4())
    return session_id


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
