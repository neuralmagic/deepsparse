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

import os

import pytest
from deepsparse.v2.text_generation import TextGenerationPipelineNoCache


@pytest.mark.parametrize(
    "onnx_model_name, raise_error",
    [("model.onnx", True), (None, True), ("model-orig.onnx", False)],
)
def test_verify_no_kv_cache_present(model_attributes, onnx_model_name, raise_error):
    _, model_path = model_attributes
    # model_path points to .../directory/model.onnx
    # we need to go up one level to .../directory
    model_path = os.path.dirname(model_path)

    if raise_error:
        with pytest.raises(ValueError):
            if onnx_model_name is None:
                TextGenerationPipelineNoCache(model_path=model_path)
            else:
                TextGenerationPipelineNoCache(
                    model_path=model_path, onnx_model_name=onnx_model_name
                )
        return
    else:
        TextGenerationPipelineNoCache(
            model_path=model_path, onnx_model_name=onnx_model_name
        )
