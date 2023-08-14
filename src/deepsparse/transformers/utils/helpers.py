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
from typing import List, Tuple, Union

import numpy
import onnx

from deepsparse.utils.onnx import translate_onnx_type_to_numpy
from sparsezoo.utils import save_onnx


__all__ = [
    "generate_session_id",
    "pad_to_fixed_length",
    "create_causal_mask",
    "overwrite_onnx_model_inputs",
]

_LOGGER = logging.getLogger(__name__)


def overwrite_onnx_model_inputs(
    onnx_file_path: str,
    sequence_length: int,
    input_ids_length: int,
    batch_size: int = 1,
) -> Tuple[str, List[int]]:
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
        -   the data type of the kv cache. If the model does not
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
        elif external_input.name.startswith("past_key_values"):
            external_input.type.tensor_type.shape.dim[2].dim_value = (
                sequence_length - input_ids_length
            )
        elif external_input.name.startswith("causal_mask"):
            external_input.type.tensor_type.shape.dim[2].dim_value = input_ids_length
            external_input.type.tensor_type.shape.dim[3].dim_value = sequence_length
        else:
            raise ValueError(f"Unexpected external input name: {external_input.name}")

    _LOGGER.info(
        "Overwriting in-place the input shapes "
        f"of the transformer model at {onnx_file_path}"
    )
    save_onnx(model, onnx_file_path)

    output_indices_to_be_cached = [
        1 if inp.name.startswith("present") else 0 for inp in model.graph.output
    ]

    kv_cache_data_type = None
    if any(output_indices_to_be_cached):
        kv_cache_elem_type = next(
            inp for inp in model.graph.input if inp.name.startswith("past_key_values")
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


def create_causal_mask(
    input_ids: Union[numpy.ndarray, List[int]],
    attention_mask: Union[numpy.ndarray, List[int]],
    dtype: numpy.dtype = numpy.int64,
) -> numpy.ndarray:
    """
    Compute a causal mask from a set of module inputs.
    In transformers, a causal mask is a boolean mask that is used to
    prevent information from future positions in a sequence from
    being used to predict the current position. Each element of the mask
    is set to 1 if the corresponding position in the input sequence
    is allowed to attend to positions up to and including that position,
    and 0 otherwise.

    in case of single-token input, the causal mask is an array
    of of shape [1, 1, 1, sequence_length],
    (essentially the reshaped attention_mask)

    in case of a multi-token input, the causal mask is an array
    of shape [batch_size, 1, input_ids_length, sequence_length]
    it is a concatenation of a:
     - past (cache) causal mask
     - and a causal mask (a lower triangular matrix of 1's and 0's)
    e.g
    ```
    input_ids = [[1,2,3,4]]
    attention_mask = [[1,1,1,1,1,1]]

    causal_mask = [[[[ 1 1 | 1 0 0 0 ],
                     [ 1 1 | 1 1 0 0 ],
                     [ 1 1 | 1 1 1 0 ],
                     [ 1 1 | 1 1 1 1 ]]]]
    ```
    or
    ```
    input_ids = [[1,2,3,4]]
    attention_mask = [[0,0,1,1,1,1,1]]

    causal_mask = [[[[ 0 0 1 1 | 1 0 0 0 ],
                     [ 0 0 1 1 | 1 1 0 0 ],
                     [ 0 0 1 1 | 1 1 1 0 ],
                     [ 0 0 1 1 | 1 1 1 1 ]]]]
    ```

    :param input_ids: input ids of the model input
    :param attention_mask: attention mask of the model input
    :param dtype: data type of the mask
    :return: causal mask
    """
    if isinstance(input_ids, numpy.ndarray):
        batch_size, input_ids_length = input_ids.shape

    else:
        batch_size, input_ids_length = 1, len(input_ids)

    if isinstance(attention_mask, numpy.ndarray):
        sequence_length = attention_mask.shape[1]
    else:
        sequence_length = len(attention_mask)
        attention_mask = numpy.array(attention_mask)[None, ...]

    if input_ids_length == 1:
        causal_mask = numpy.reshape(attention_mask, (batch_size, 1, 1, sequence_length))
        return causal_mask.astype(dtype)

    causal_mask = numpy.tril(
        numpy.ones((batch_size, 1, input_ids_length, input_ids_length), dtype=dtype), 0
    )
    past_causal_mask = numpy.ones(
        (batch_size, 1, input_ids_length, sequence_length - input_ids_length),
        dtype=dtype,
    )
    causal_mask = numpy.concatenate((past_causal_mask, causal_mask), axis=-1)

    num_zeros = numpy.count_nonzero(attention_mask == 0)

    # zero out the dimensions that correspond to tokens that we do not
    # want to attend to
    causal_mask[:, :, :, :num_zeros] = 0

    return causal_mask
