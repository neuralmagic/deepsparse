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
import pathlib
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy
from transformers import AutoTokenizer, GenerationConfig

from deepsparse.utils.onnx import CACHE_INPUT_PREFIX, CACHE_OUTPUT_PREFIX


__all__ = [
    "generate_session_id",
    "pad_to_fixed_length",
    "create_causal_mask",
    "repeat_inputs",
    "initialize_kv_cache_state",
    "prepends_bos_token",
    "check_and_return_generation_config",
    "override_config",
    "process_generation_config",
    "validate_session_ids",
    "compute_engine_inputs",
    "set_generated_length",
]

_LOGGER = logging.getLogger(__name__)


def set_generated_length(
    max_length: int,
    prompt_tokens_length: int,
    sequence_length: int,
    prompt_sequence_length: int,
    max_new_tokens: int,
    finish_reason_choices: "FinishReason",  # noqa
):
    """
    Determine the length of the generated tokens. The hard cap on the total number
    of tokens is based on the sequence length. If max_length is provided and is less
    than the sequence length, it will be used to cap the total number of tokens
    generated. If it is not provided, the max_new_tokens attribute will be used and also
    capped by the sequence length.

    :param max_length: max_length attribute, provided as input during inference
    :param prompt_tokens_length: the number of prompt tokens used as part of the
        generated output
    :param sequence_length: the sequence length used for the pipeline
    :param prompt_sequence_length: the prompt sequence length used for the pipeline
    :param max_new_tokens: the max_new_tokens attribute, which may be provided
    as part of the input during inference
    """
    if max_length:
        # if max_length provided, use that to cap total tokens generated
        max_tokens = max_length
        finish_reason = finish_reason_choices.LENGTH
    else:
        # if not provided, max tokens is based on max_new_tokens + prompt tokens
        max_tokens = (
            min(max_new_tokens, sequence_length - prompt_sequence_length)
            + prompt_tokens_length
        )
        finish_reason = finish_reason_choices.MAX_NEW_TOKENS

    # hard model/pipeline cap
    return (
        (sequence_length, finish_reason_choices.CAPACITY)
        if sequence_length < max_tokens
        else (max_tokens, finish_reason)
    )


def compute_engine_inputs(onnx_input_names: str, **kwargs) -> List[numpy.ndarray]:
    """
    Given the names of the onnx inputs, compute the inputs
    to the engine. The inputs will be calculating from the
    passed kwargs. The information about the required kwargs
    can be found in the docstring of the individual compute
    functions.

    :param onnx_input_names: The names of the onnx inputs
    :param kwargs: The kwargs to compute the inputs from
    :return: The computed inputs to the engine
    """
    engine_inputs = []
    for input_name in onnx_input_names:
        if input_name == "causal_mask":
            # delay the computation of the causal mask
            continue
        # fetch the compute function for the
        # given input_name
        compute_func = _get_compute_func(input_name)
        # compute the engine input from the kwargs
        # and append it to the engine_inputs
        engine_inputs.append(compute_func(**kwargs))

    if "causal_mask" in onnx_input_names:
        # compute the causal mask and append it to the engine_inputs
        input_ids, attention_mask, *_ = engine_inputs
        engine_inputs.append(create_causal_mask(input_ids, attention_mask))

    return engine_inputs


def _get_compute_func(input_name: str) -> Callable[..., numpy.ndarray]:
    # given the input_name, return the appropriate compute function
    compute_func = {
        "input_ids": _compute_input_ids,
        "attention_mask": _compute_attention_mask,
        "positions": _compute_positions,
    }.get(input_name)
    if compute_func is None:
        raise ValueError(
            "Could not find compute function " f"for the input_name: {input_name}"
        )
    return compute_func


def _compute_input_ids(token_batch: List[int], **kwargs) -> numpy.ndarray:
    # convert the token_batch to a numpy array
    return numpy.array([token_batch])


def _compute_attention_mask(
    sequence_length: int,
    prompt_sequence_length: int,
    num_total_processed_tokens: int,
    **kwargs,
) -> numpy.ndarray:
    # create a fully masked attention mask with the appropriate
    # shape (equal to the sequence_length)
    attention_mask = numpy.zeros((1, sequence_length), dtype=numpy.int64)
    # unmask the appropriate number of tokens, the sum of
    # - the number of tokens already processed and cached (num_total_processed_tokens)
    # - the number of tokens currently processed (prompt_sequence_length)
    # the sum cannot exceed the maximum length of the attention_mask
    num_attention_entries_to_unmask = min(
        num_total_processed_tokens + prompt_sequence_length, sequence_length
    )
    # unmask the bits from the right-hand side
    attention_mask[:, -num_attention_entries_to_unmask:] = 1
    return attention_mask


def _compute_positions(
    num_total_processed_tokens: int, prompt_sequence_length: int, **kwargs
):
    # create the positions array with the appropriate shape
    # positions count starts from the number of tokens already processed
    # and ends at the number of tokens already processed + the number of tokens
    # currently processed
    return (
        numpy.arange(
            num_total_processed_tokens,
            num_total_processed_tokens + prompt_sequence_length,
        )
        .reshape(1, -1)
        .astype(numpy.int64)
    )


def validate_session_ids(
    session_ids: Optional[str], other_attributes: Dict[str, Any]
) -> Optional[List[str]]:
    """
    Helper function to validate the session ids for TextGenerationInput schema

    :param session_ids: The session ids to validate
    :param other_attributes: The other attributes of the input schema
    :return: The session ids if they were not None in the
        first place, otherwise None
    """
    if session_ids is None:
        return None

    if not isinstance(session_ids, list):
        session_ids = [session_ids]

    if isinstance(other_attributes["sequences"], str) and len(session_ids) != 1:
        raise ValueError(
            f"Only one session id is allowed for a single input sequence. "
            f"Detected 1 input sequence and {len(session_ids)} session ids"
        )
    if isinstance(other_attributes["sequences"], list) and len(session_ids) != len(
        other_attributes["sequences"]
    ):
        raise ValueError(
            f"Number of session ids must match the number of input sequences. "
            f"Detected {len(other_attributes['sequences'])} "
            f"input sequences and {len(session_ids)} session ids"
        )
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(
            f"Session ids must be unique. Detected session_ids: {session_ids}"
        )

    return session_ids


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
    kv_cache_data_type: numpy.dtype,
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

    :return: dictionary of kv cache tensors, where the keys are the
        output names of the kv cache tensors and the values are the
        kv cache tensors of shape
        (batch_size, num_attention_heads, length, hidden_dims)
    """
    batch_size, num_attention_heads, length_, hidden_dims = cache_shape

    # new kv cache tensor is either
    # - non-empty tensor of zeros with shape
    #   (batch_size, num_attention_heads, length, hidden_dims),
    #   required for the external kv cache management
    # or
    # - empty tensor with shape
    #   (0, num_attention_heads, length, hidden_dims)
    #   required for the internal kv cache management
    kv_cache_tensor = numpy.zeros(
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
    return {key: kv_cache_tensor for key in cache_keys}


def generate_session_id() -> str:
    """
    Generate uuid for session id. This is used to
    identify the kv cache session for the user
    """
    session_id = str(uuid.uuid4())
    return session_id


def process_generation_config(
    generation_config: Union[None, str, pathlib.Path, Dict, GenerationConfig]
) -> Union[GenerationConfig, None]:
    """
    Process and return a GenerationConfig. The function can take in a filepath
    pointing to a json consisting of the config values, a dictionary with the config
    values, or a loaded GenerationConfig object. If None is given, the defaults are,
    the pipeline GenerationConfig is used, if provided. If both are None, default
    are used for generation.

    :param generation_config: either a json filepath, dictionary or loaded
    GenerationConfig object

    :return: GenerationConfig object or None

    """
    if isinstance(generation_config, GenerationConfig):
        return generation_config

    if not generation_config:
        return None

    if isinstance(generation_config, dict):
        return GenerationConfig.from_dict(generation_config)

    if isinstance(generation_config, (str, pathlib.Path)):
        generation_config = pathlib.Path(generation_config)
        config_dir = generation_config.parent.absolute()
        config_name = generation_config.name

    generation_config = GenerationConfig.from_pretrained(config_dir, config_name)
    return generation_config


def check_and_return_generation_config(
    pipeline_generation_config: [None, str, pathlib.Path, Dict, GenerationConfig],
    input_generation_config: [None, str, pathlib.Path, Dict, GenerationConfig],
    defaults: "GenerationDefaults",  # noqa F821
) -> Union[GenerationConfig, None]:
    """
    Check if an input generation config is provided. If not, check if a pipeline
    generation config exists. If neither exists, use the defualt generation configs,
    either deespsparse defaults or hugging face defaults. If a pipeline config exists
    and an input config exists, use the input config.

    :param pipeline_generation_config: either a json filepath, dictionary or loaded
    GenerationConfig object provided by the user during pipeline creation
    :param input_generation_config: either a json filepath, dictionary or loaded
    GenerationConfig object provided by the user during inference
    :param defaults: defaults to use for the GenerationConfig if a config is not
    provided during inference or pipeline creation.

    :return: GenerationConfig object or None

    """
    generation_config = process_generation_config(input_generation_config)
    if generation_config is None:
        if pipeline_generation_config:
            generation_config = pipeline_generation_config
    else:
        _LOGGER.debug(
            "Input generation config detected. This will override any"
            " config provided during pipeline creation."
        )

    if not generation_config:
        _LOGGER.debug("No GenerationConfig detected. Using GenerationDefaults values")
        generation_config = defaults
    return generation_config


def override_config(
    overrides: Optional[Dict], generation_config: GenerationConfig
) -> GenerationConfig:
    """
    Override any generation config properties using the `kwargs` argument in
    TextGenerationInput. If None, the generation config is returned unchanged.
    If provided, update all attribute stored in the dictionary. An errror will be
    raised if the dictionary contains an key which is not a GenerationConfig
    attribute.

    :param overrides: dictionary containing GenerationConfig attributes to update
    :param generation_config: GenerationConfig to update

    :return: GenerationConfig object

    """
    if overrides is None:
        return generation_config

    for k, v in overrides.items():
        if hasattr(generation_config, k):
            setattr(generation_config, k, v)
            _LOGGER.debug(f"Overriding attribute {k} in the generation config")
        else:
            raise AttributeError(
                f"Argument {k} provided for GenerationConfig is not "
                "valid. Refer to the TextGenerationInput for supported attributes. "
            )

    return generation_config


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
    The padding is done on the left side of the array.

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
    padding[axis] = (max_len - array.shape[axis], 0)
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
