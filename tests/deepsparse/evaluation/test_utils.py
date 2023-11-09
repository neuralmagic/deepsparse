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

from transformers import GPTNeoForCausalLM

import pytest
from src.deepsparse.evaluation.utils import initialize_model_from_target


@pytest.fixture
def pipeline_target():
    return "hf:mgoin/TinyStories-1M-deepsparse"


@pytest.fixture
def torch_target():
    return "roneneldan/TinyStories-1M"


def test_initialize_model_from_target_pipeline_onnx(pipeline_target):
    model = initialize_model_from_target(pipeline_target, "onnxruntime")
    assert model.engine_type == "onnxruntime"


def test_initialize_model_from_target_pipeline_deepsparse(pipeline_target):
    model = initialize_model_from_target(pipeline_target, "deepsparse")
    assert model.engine_type == "deepsparse"


def test_initialize_model_from_target_torch(torch_target):
    model = initialize_model_from_target(torch_target, "torch")
    assert isinstance(model, GPTNeoForCausalLM)
