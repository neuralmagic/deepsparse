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

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GPTNeoForCausalLM,
)

import pytest
from deepsparse import Pipeline
from src.deepsparse.evaluation.utils import (
    create_model_from_target,
    get_save_path,
    is_model_llm,
    resolve_integration,
)


@pytest.fixture
def llm_type_hf_model():
    return AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")


@pytest.fixture
def not_llm_type_hf_model():
    return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")


@pytest.fixture
def llm_type_pipeline():
    return Pipeline.create(
        task="text-generation",
        model_path="hf:mgoin/TinyStories-1M-deepsparse",
        engine_type="onnxruntime",
    )


def test_resolve_known_llm_model(llm_type_hf_model):
    assert (
        resolve_integration(model=llm_type_hf_model, datasets="")
        == "llm-evaluation-harness"
    )


def test_resolve_unknown_model(not_llm_type_hf_model):
    assert resolve_integration(model=not_llm_type_hf_model, datasets="") is None


def test_is_model_llm_hf_true(llm_type_hf_model):
    assert is_model_llm(llm_type_hf_model)


def test_is_model_llm_hf_false(not_llm_type_hf_model):
    assert not is_model_llm(not_llm_type_hf_model)


def test_is_model_llm_pipeline_true(llm_type_pipeline):
    assert is_model_llm(llm_type_pipeline)


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
    model = create_model_from_target(pipeline_target, "onnxruntime")
    assert model.ops.get("single_engine")._engine_type == "onnxruntime"


def test_initialize_model_from_target_pipeline_deepsparse(pipeline_target):
    model = create_model_from_target(pipeline_target, "deepsparse")
    assert model.ops.get("single_engine")._engine_type == "deepsparse"


def test_initialize_model_from_target_pipeline_with_kwargs(pipeline_target):
    model = create_model_from_target(pipeline_target, "deepsparse", sequence_length=64)
    assert model.ops.get("process_input").sequence_length == 64


def test_initialize_model_from_target_torch(torch_target):
    model = create_model_from_target(torch_target, "torch")
    assert isinstance(model, GPTNeoForCausalLM)
