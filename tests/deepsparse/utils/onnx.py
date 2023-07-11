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


import onnx

import pytest
from deepsparse.utils import override_onnx_batch_size, override_onnx_input_shapes
from sparsezoo import Model


@pytest.mark.parametrize(
    "test_model, batch_size",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",  # noqa: E501
            10,
        )
    ],
    scope="function",
)
@pytest.mark.parametrize("inplace", [True, False], scope="function")
def test_override_onnx_batch_size(test_model, batch_size, inplace):
    onnx_file_path = Model(test_model).onnx_model.path
    # Override the batch size of the ONNX model
    with override_onnx_batch_size(
        onnx_file_path, batch_size, inplace=inplace
    ) as modified_model_path:
        # Load the modified ONNX model
        modified_model = onnx.load(modified_model_path)
        assert (
            modified_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
            == batch_size
        )


@pytest.mark.parametrize(
    "test_model, input_shapes",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",  # noqa: E501
            [10, 224, 224, 3],
        )
    ],
    scope="function",
)
@pytest.mark.parametrize("inplace", [True, False], scope="function")
def test_override_onnx_input_shapes(test_model, input_shapes, inplace):
    onnx_file_path = Model(test_model).onnx_model.path
    # Override the batch size of the ONNX model
    with override_onnx_input_shapes(
        onnx_file_path, input_shapes, inplace=inplace
    ) as modified_model_path:
        # Load the modified ONNX model
        modified_model = onnx.load(modified_model_path)
        new_input_shapes = [
            dim.dim_value
            for dim in modified_model.graph.input[0].type.tensor_type.shape.dim
        ]
        assert new_input_shapes == input_shapes
