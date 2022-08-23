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
from deepsparse.server_v2.config import (
    EndpointConfig,
    ImageSizesConfig,
    SequenceLengthsConfig,
)


def test_no_bucketing_config():
    cfg = EndpointConfig(task="", model="").to_pipeline_config()
    assert cfg.input_shapes is None
    assert cfg.kwargs == {}


@pytest.mark.parametrize("task", ["yolo", "yolact", "image_classification"])
def test_bucketing_sequence_length_for_cv(task):
    with pytest.raises(ValueError, match=f"for non-nlp task {task}"):
        EndpointConfig(
            task=task, model="", bucketing=SequenceLengthsConfig(sequence_lengths=[])
        ).to_pipeline_config()


@pytest.mark.parametrize(
    "task", ["question_answering", "text_classification", "token_classification"]
)
def test_bucketing_image_size_for_nlp(task):
    with pytest.raises(ValueError, match=f"for non computer vision task {task}"):
        EndpointConfig(
            task=task, model="", bucketing=ImageSizesConfig(image_sizes=[])
        ).to_pipeline_config()


def test_bucketing_zero_sequence_length():
    with pytest.raises(ValueError, match="at least one sequence length"):
        EndpointConfig(
            task="qa", model="", bucketing=SequenceLengthsConfig(sequence_lengths=[])
        ).to_pipeline_config()


def test_bucketing_zero_image_size():
    with pytest.raises(ValueError, match="at least one image size"):
        EndpointConfig(
            task="yolo", model="", bucketing=ImageSizesConfig(image_sizes=[])
        ).to_pipeline_config()


def test_bucketing_one_sequence_length():
    cfg = EndpointConfig(
        task="qa", model="", bucketing=SequenceLengthsConfig(sequence_lengths=[32])
    ).to_pipeline_config()
    assert cfg.input_shapes is None
    assert cfg.kwargs == {"sequence_length": 32}


def test_bucketing_multi_sequence_length():
    cfg = EndpointConfig(
        task="qa", model="", bucketing=SequenceLengthsConfig(sequence_lengths=[32, 64])
    ).to_pipeline_config()
    assert cfg.input_shapes is None
    assert cfg.kwargs == {"sequence_length": [32, 64]}


def test_bucketing_one_image_size():
    cfg = EndpointConfig(
        task="yolo", model="", bucketing=ImageSizesConfig(image_sizes=[(256, 256)])
    ).to_pipeline_config()
    assert cfg.input_shapes == [[256, 256]]
    assert cfg.kwargs == {}


def test_endpoint_config_to_pipeline_copy_fields():
    cfg = EndpointConfig(task="qa", model="zxcv").to_pipeline_config()
    assert cfg.task == "qa"
    assert cfg.model_path == "zxcv"
    assert cfg.batch_size is None  # dynamic batch by default


def test_endpoint_config_to_pipeline_config_static_batch():
    cfg = EndpointConfig(
        task="", model="", accept_multiples_of_batch_size=False
    ).to_pipeline_config()
    assert cfg.batch_size == 1

    cfg = EndpointConfig(
        task="", model="", batch_size=64, accept_multiples_of_batch_size=False
    ).to_pipeline_config()
    assert cfg.batch_size == 64
