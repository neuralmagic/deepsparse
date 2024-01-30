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
from deepsparse.evaluation.utils import (
    create_pipeline,
    get_save_path,
    if_generative_language_model,
    resolve_integration,
)


@pytest.fixture
def llm_type_pipeline():
    return Pipeline.create(
        task="text-generation",
        model_path="hf:mgoin/TinyStories-1M-deepsparse",
        engine_type="onnxruntime",
    )


def test_resolve_known_llm_pipeline(llm_type_pipeline):
    assert (
        resolve_integration(pipeline=llm_type_pipeline, datasets="")
        == "lm-evaluation-harness"
    )


def test_if_generative_language_pipeline_true(llm_type_pipeline):
    assert if_generative_language_model(llm_type_pipeline)


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


def test_initialize_model_from_target_pipeline_onnx(pipeline_target):
    model = create_pipeline(pipeline_target, "onnxruntime")
    assert model.ops.get("single_engine")._engine_type == "onnxruntime"


def test_initialize_model_from_target_pipeline_with_kwargs(pipeline_target):
    model = create_pipeline(pipeline_target, "deepsparse", sequence_length=64)
    assert model.ops.get("process_input").sequence_length == 64
