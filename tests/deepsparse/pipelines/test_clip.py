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
    CLIPCaptionInput,
    CLIPCaptionPipeline,
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
    model_path = None
    return CLIPVisualInput(images=images.get("images")), model_path


@pytest.fixture
def text_input():
    model_path = None
    text = ["a building", "a dog", "a cat"]
    return CLIPTextInput(text=text), model_path


@pytest.mark.skip(reason="No CLIP models currently available to run tests")
@mock_engine(rng_seed=0)
def test_visual_clip(engine, visual_input):
    model_path = visual_input[-1]
    pipeline = Pipeline.create(task="clip_visual", model_path=model_path)
    assert isinstance(pipeline, CLIPVisualPipeline)
    output = pipeline(visual_input[0])
    assert isinstance(output, CLIPVisualOutput)
    assert len(output.image_embeddings) == 1


@pytest.mark.skip(reason="No CLIP models curently available to run tests")
@mock_engine(rng_seed=0)
def test_text_clip(engine, text_input):
    model_path = text_input[-1]
    pipeline = Pipeline.create(task="clip_text", model_path=model_path)
    assert isinstance(pipeline, CLIPTextPipeline)
    output = pipeline(text_input[0])
    assert isinstance(output, CLIPTextOutput)
    assert len(output.text_embeddings) == 1


@pytest.mark.skip(reason="No CLIP models currently available to run tests")
@mock_engine(rng_seed=0)
def test_zero_shot(engine, visual_input, text_input):
    model_path_text = text_input[-1]
    model_path_visual = visual_input[-1]
<<<<<<< HEAD
    kwargs = {
        "visual_model_path": model_path_visual,
        "text_model_path": model_path_text,
    }
=======
    visual_args = {"model_path": model_path_visual}
    text_args = {"model_path": model_path_text}
    kwargs = {"visual_args": visual_args, "text_args": text_args}
>>>>>>> move paths to fixtures
    pipeline = BasePipeline.create(task="clip_zeroshot", **kwargs)
    assert isinstance(pipeline, CLIPZeroShotPipeline)
    pipeline_input = CLIPZeroShotInput(
        image=CLIPVisualInput(images=visual_input[0].images[-1]), text=text_input[0]
    )
    output = pipeline(pipeline_input)
    assert isinstance(output, CLIPZeroShotOutput)


@pytest.mark.skip(reason="No CLIP models currently available to run tests")
@mock_engine(rng_seed=0)
def test_caption(engine, visual_input, text_input):
    model_path_visual = text_input[-1]
    model_path_text = text_input[-1]
    model_path_decoder = None
    pipeline_input = CLIPCaptionInput(
        image=CLIPVisualInput(images=visual_input[0].images[-1])
    )
    kwargs = {
        "visual_model_path": model_path_visual,
        "text_model_path": model_path_text,
        "decoder_model_path": model_path_decoder,
    }
    pipeline = BasePipeline.create(task="clip_caption", **kwargs)
    assert isinstance(pipeline, CLIPCaptionPipeline)
    assert isinstance(pipeline_input, CLIPCaptionInput)
