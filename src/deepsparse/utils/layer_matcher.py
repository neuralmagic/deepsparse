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

from typing import Dict

import numpy
import onnx
from onnx import numpy_helper

import torch


__all__ = [
    "onnx_torch_matcher",
]


OP_TYPES = ["Conv", "MatMul", "Gemm", "MatMulInteger", "ConvInteger"]
QUANTIZED_LINEAR_OP_TYPES = ["QLinearConv", "QLinearMatMul"]

EPSILON = 1e-5


def onnx_max_value_to_name(onnx_model_path: str) -> Dict[float, str]:
    """
    Map the max onnx node value to the name of the node

    :param onnx_model_path: path to .onnx
    """
    onnx_model = onnx.load(onnx_model_path)
    onnx_weight_names = [
        node.input[1] for node in onnx_model.graph.node if node.op_type in OP_TYPES
    ]
    onnx_weight_names.extend(
        [
            node.input[3]
            for node in onnx_model.graph.node
            if node.op_type in QUANTIZED_LINEAR_OP_TYPES
        ]
    )

    onnx_mapper = {}
    for init in onnx_model.graph.initializer:
        if init.name in onnx_weight_names:
            arr_max = numpy.max(numpy_helper.to_array(init))
            onnx_mapper[arr_max] = init.name

    return onnx_mapper


def torch_max_values_to_name(torch_model_path: str) -> Dict[float, str]:
    """
    Map the max torch node value to the name of the node

    :param onnx_model_path: path to .onnx
    """
    torch_model = torch.load(torch_model_path, map_location=torch.device("cpu"))
    torch_mapper = {}

    if "state_dict" in torch_model:
        torch_model = torch_model["state_dict"]

    for key, val in torch_model.items():
        torch_mapper[numpy.max(val.numpy())] = key

    return torch_mapper


def onnx_torch_matcher(
    onnx_model_path: str, torch_model_path: str, epsilon: float = EPSILON
) -> Dict[str, str]:
    """
    Match the onnx init name to torch names as a dictionary. Dict keys
    will be one of Conv, MatMul, Gemm, MatMulInteger, ConvInteger,
    QLinearConv and QLinearMatMul.

    Layer matching based on the max value in the array within +/- eplison

    :param onnx_model_path: path to .onnx
    :param torch_model_path: path to .pth
    """

    onnx_max_values: Dict[float, str] = onnx_max_value_to_name(onnx_model_path)
    torch_max_values: Dict[float, str] = torch_max_values_to_name(torch_model_path)

    onnx_torch_matches: Dict[str, str] = {}
    for key_onnx in onnx_max_values:
        for key_torch in torch_max_values:
            if abs(key_onnx - key_torch) <= epsilon:
                onnx_torch_matches[onnx_max_values[key_onnx]] = torch_max_values[
                    key_torch
                ]

    return onnx_torch_matches
