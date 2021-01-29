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

import contextlib
import os
import tempfile
from typing import List

import numpy
import onnx

from deepsparse.utils.log import log_init


__all__ = [
    "get_external_inputs",
    "get_external_outputs",
    "get_input_names",
    "get_output_names",
    "generate_random_inputs",
    "override_onnx_batch_size",
]

log = log_init(os.path.basename(__file__))

onnx_tensor_type_map = {
    1: numpy.float32,
    2: numpy.uint8,
    3: numpy.int8,
    4: numpy.uint16,
    5: numpy.int16,
    6: numpy.int32,
    7: numpy.int64,
    9: numpy.bool_,
    10: numpy.float16,
    11: numpy.float64,
    12: numpy.uint32,
    13: numpy.uint64,
    14: numpy.complex64,
    15: numpy.complex128,
}


def translate_onnx_type_to_numpy(tensor_type: int):
    """
    Translates ONNX types to numpy types
    :param tensor_type: Integer representing a type in ONNX spec
    :return: Corresponding numpy type
    """
    if tensor_type not in onnx_tensor_type_map:
        raise Exception("Unknown ONNX tensor type = {}".format(tensor_type))
    return onnx_tensor_type_map[tensor_type]


def get_external_inputs(onnx_filepath: str) -> List:
    """
    Gather external inputs of ONNX model
    :param onnx_filepath: File path to ONNX model
    :return: List of input objects
    """
    model = onnx.load(onnx_filepath)
    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    external_inputs = [
        input for input in all_inputs if input.name not in initializer_input_names
    ]
    return external_inputs


def get_external_outputs(onnx_filepath: str) -> List:
    """
    Gather external outputs of ONNX model
    :param onnx_filepath: File path to ONNX model
    :return: List of output objects
    """
    model = onnx.load(onnx_filepath)
    return [output for output in model.graph.output]


def get_input_names(onnx_filepath: str) -> List[str]:
    """
    Gather names of all external inputs of ONNX model
    :param onnx_filepath: File path to ONNX model
    :return: List of string names
    """
    return [input.name for input in get_external_inputs(onnx_filepath)]


def get_output_names(onnx_filepath: str) -> List[str]:
    """
    Gather names of all external outputs of ONNX model
    :param onnx_filepath: File path to ONNX model
    :return: List of string names
    """
    return [output.name for output in get_external_outputs(onnx_filepath)]


def generate_random_inputs(
    onnx_filepath: str, batch_size: int = None
) -> List[numpy.array]:
    """
    Generate random data that matches the type and shape of ONNX model,
    with a batch size override
    :param onnx_filepath: File path to ONNX model
    :param batch_size: If provided, override for the batch size dimension
    :return: List of random tensors
    """
    input_data_list = []
    for i, external_input in enumerate(get_external_inputs(onnx_filepath)):
        input_tensor_type = external_input.type.tensor_type
        in_shape = [int(d.dim_value) for d in input_tensor_type.shape.dim]

        if batch_size is not None:
            in_shape[0] = batch_size

        log.info("-- generating random input #{} of shape = {}".format(i, in_shape))
        input_data_list.append(
            numpy.random.rand(*in_shape).astype(
                translate_onnx_type_to_numpy(input_tensor_type.elem_type)
            )
        )
    return input_data_list


@contextlib.contextmanager
def override_onnx_batch_size(onnx_filepath: str, batch_size: int):
    """
    Rewrite batch sizes of ONNX model, saving the modified model and returning its path
    :param onnx_filepath: File path to ONNX model
    :param batch_size: Override for the batch size dimension
    :return: File path to modified ONNX model
    """
    model = onnx.load(onnx_filepath)
    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    external_inputs = [
        input for input in all_inputs if input.name not in initializer_input_names
    ]
    for external_input in external_inputs:
        external_input.type.tensor_type.shape.dim[0].dim_value = batch_size

    # Save modified model
    shaped_model = tempfile.NamedTemporaryFile(mode="w", delete=False)
    onnx.save(model, shaped_model.name)

    try:
        yield shaped_model.name
    finally:
        os.unlink(shaped_model.name)
        shaped_model.close()
