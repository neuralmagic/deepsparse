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

from transformers import GPTNeoForCausalLM

import pytest
from src.deepsparse.evaluation.utils import (
    get_save_path,
    text_generation_model_from_target,
)


def test_get_save_path_path_provided(tmpdir):
    save_path = get_save_path(
        type_serialization="json", save_path=tmpdir, file_name="dummy"
    )
    assert save_path == os.path.join(tmpdir, "dummy.json")


def test_get_save_to_current_working_directory():
    save_path = get_save_path(type_serialization="json", file_name="dummy")
    assert save_path == os.path.join(os.getcwd(), "dummy.json")


@pytest.fixture
def pipeline_target():
    return "hf:mgoin/TinyStories-1M-deepsparse"


@pytest.fixture
def torch_target():
    return "roneneldan/TinyStories-1M"


def test_initialize_model_from_target_pipeline_onnx(pipeline_target):
    model = text_generation_model_from_target(pipeline_target, "onnxruntime")
    assert model.ops.get("single_engine")._engine_type == "onnxruntime"


def test_initialize_model_from_target_pipeline_deepsparse(pipeline_target):
    model = text_generation_model_from_target(pipeline_target, "deepsparse")
    assert model.ops.get("single_engine")._engine_type == "deepsparse"


def test_initialize_model_from_target_torch(torch_target):
    model = text_generation_model_from_target(torch_target, "torch")
    assert isinstance(model, GPTNeoForCausalLM)
