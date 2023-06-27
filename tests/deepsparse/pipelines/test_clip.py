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

import pytest
from deepsparse import BasePipeline, Pipeline
from deepsparse.clip import (
    CLIPTextInput,
    CLIPTextOutput,
    CLIPTextPipeline,
    CLIPVisualInput,
    CLIPVisualOutput,
    CLIPVisualPipeline,
    CLIPZeroShotInput,
    CLIPZeroShotOutput,
    CLIPZeroShotPipeline,
)
from tests.deepsparse.pipelines.data_helpers import computer_vision
from tests.utils import mock_engine


@pytest.fixture
def visual_input():
    images = computer_vision(batch_size=2)
    return CLIPVisualInput(images=images.get("images"))


@pytest.fixture
def text_input():
    text = ["a building", "a dog", "a cat"]
    return CLIPTextInput(text=text)


@mock_engine(rng_seed=0)
def test_visual_clip(engine, visual_input):
    model_path = "clip_onnx/clip_visual.onnx"
    pipeline = Pipeline.create(task="clip_visual", model_path=model_path)
    assert isinstance(pipeline, CLIPVisualPipeline)
    output = pipeline(visual_input)
    assert isinstance(output, CLIPVisualOutput)
    assert len(output.image_embeddings) == 1


@mock_engine(rng_seed=0)
def test_text_clip(engine, text_input):
    model_path = "clip_onnx/clip_text.onnx"
    pipeline = Pipeline.create(task="clip_text", model_path=model_path)
    assert isinstance(pipeline, CLIPTextPipeline)
    output = pipeline(text_input)
    assert isinstance(output, CLIPTextOutput)
    assert len(output.text_embeddings) == 1


@mock_engine(rng_seed=0)
def test_zero_shot(engine, visual_input, text_input):
    model_path_text = "clip_onnx/clip_text.onnx"
    model_path_visual = "clip_onnx/clip_visual.onnx"
    visual_args = {"model_path": model_path_visual}
    text_args = {"model_path": model_path_text}
    kwargs = {"visual_args": visual_args, "text_args": text_args}
    pipeline = BasePipeline.create(task="clip_zeroshot", **kwargs)
    assert isinstance(pipeline, CLIPZeroShotPipeline)
    pipeline_input = CLIPZeroShotInput(
        image=CLIPVisualInput(images=visual_input.images[-1]), text=text_input
    )
    output = pipeline(pipeline_input)
    assert isinstance(output, CLIPZeroShotOutput)
