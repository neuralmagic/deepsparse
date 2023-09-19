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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
from transformers import AutoTokenizer

from deepsparse.utils.onnx import CACHE_INPUT_PREFIX, CACHE_OUTPUT_PREFIX


__all__ = [
    "generate_session_id",
    "pad_to_fixed_length",
    "create_causal_mask",
    "repeat_inputs",
    "initialize_kv_cache_state",
    "prepends_bos_token",
]

_LOGGER = logging.getLogger(__name__)


def prepends_bos_token(tokenizer: AutoTokenizer) -> bool:
    """
    Check whether the tokenizer prepends a BOS token to the input sequence.

    :param tokenizer: tokenizer to check
    :return: True if the tokenizer prepends a BOS token to the input sequence,
        False otherwise
    """
    if hasattr(tokenizer, "add_bos_token"):
        return bool(tokenizer.add_bos_token)
    return False


def initialize_kv_cache_state(
    cache_shape: Tuple[int, int, int, int],
    kv_cache_data_type: Any,  # TODO: add type
    output_names: List[str],
    length: Optional[int] = None,
    empty: bool = False,
) -> Dict[str, numpy.ndarray]:
    """
    Initialize the kv cache state for the given set of arguments.

    :param cache_shape: shape of the kv cache tensor. Should be
        (batch_size, num_attention_heads, length, hidden_dims)
    :param kv_cache_data_type: data type of the kv cache tensor
    :param output_names: list of output names from the engine
    :param length: length of the input sequence. If None, the length
        is taken from the cache_shape
    :param empty: if True, initialize an empty kv cache tensor
        with batch_size set to 0. Otherwise, initialize a kv cache
        tensor with zeros
    """
    batch_size, num_attention_heads, length_, hidden_dims = cache_shape

    empty_kv_cache_tensor = numpy.zeros(
        (
            batch_size if not empty else 0,
            num_attention_heads,
            length if length is not None else length_,
            hidden_dims,
        ),
        dtype=kv_cache_data_type,
    )

    cache_keys = [
        output_name.replace(CACHE_OUTPUT_PREFIX, CACHE_INPUT_PREFIX)
        for output_name in output_names
        if output_name.startswith(CACHE_OUTPUT_PREFIX)
    ]
    return {key: empty_kv_cache_tensor for key in cache_keys}


def generate_session_id() -> str:
    """
    Generate uuid for session id. This is used to
    identify the kv cache session for the user
    """
    session_id = str(uuid.uuid4())
    return session_id


def repeat_inputs(
    input_sequences: List[str], num_generated_predictions: int
) -> List[str]:
    """
    :param input_sequences: List of input sequences to repeat
    :param num_generated_predictions: number of times to repeat each sequence

    :return: a list of input sequences, where sequences have been repeated
        num_generated_predictions times if the sequence appears in input_sequences just
        once. If the sequence appears multiple times in input_sequences, the
        num_generated_predictions for the sequence is ignored.
    """
    repeated_seq = []

    for seq in input_sequences:
        repeated_seq.extend(numpy.repeat([seq], num_generated_predictions))
    return repeated_seq


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
