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
from deepsparse.engine import Scheduler
from deepsparse.server_v2.config import (
    EndpointConfig,
    ImageSizesConfig,
    SequenceLengthsConfig,
    ServerConfig,
)
from deepsparse.server_v2.main import (
    _endpoint_config_to_pipeline_config,
    _get_scheduler,
    _unpack_bucketing,
    _unpack_cv_bucketing,
    _unpack_nlp_bucketing,
)


def test_scheduler():
    assert (
        _get_scheduler(
            ServerConfig(num_cores=10, num_concurrent_batches=1, endpoints=[])
        )
        == Scheduler.single_stream
    )
    assert (
        _get_scheduler(
            ServerConfig(num_cores=10, num_concurrent_batches=10, endpoints=[])
        )
        == Scheduler.multi_stream
    )


def test_no_bucketing_config():
    assert _unpack_bucketing("", None) == (None, {})


@pytest.mark.parametrize("task", ["yolo", "yolact", "image_classification"])
def test_bucketing_sequence_length_for_cv(task):
    with pytest.raises(ValueError, match=f"for non-nlp task {task}"):
        _unpack_bucketing(task, SequenceLengthsConfig(sequence_lengths=[]))


@pytest.mark.parametrize(
    "task", ["question_answering", "text_classification", "token_classification"]
)
def test_bucketing_image_size_for_nlp(task):
    with pytest.raises(ValueError, match=f"for non computer vision task {task}"):
        _unpack_bucketing(task, ImageSizesConfig(image_sizes=[]))


def test_bucketing_zero_sequence_length():
    with pytest.raises(ValueError, match="at least one sequence length"):
        _unpack_nlp_bucketing(SequenceLengthsConfig(sequence_lengths=[]))


def test_bucketing_one_sequence_length():
    assert (None, {"sequence_length": 32}) == _unpack_nlp_bucketing(
        SequenceLengthsConfig(sequence_lengths=[32])
    )


def test_bucketing_multi_sequence_length():
    assert (None, {"sequence_length": [32, 64]}) == _unpack_nlp_bucketing(
        SequenceLengthsConfig(sequence_lengths=[32, 64])
    )


def test_bucketing_zero_image_size():
    with pytest.raises(ValueError, match="at least one image size"):
        _unpack_cv_bucketing(ImageSizesConfig(image_sizes=[]))


def test_bucketing_one_image_size():
    assert ([[256, 256]], {}) == _unpack_cv_bucketing(
        ImageSizesConfig(image_sizes=[(256, 256)])
    )


def test_bucketing_multi_image_size():
    with pytest.raises(NotImplementedError):
        _unpack_cv_bucketing(ImageSizesConfig(image_sizes=[(256, 256), (512, 512)]))


def _model_helper(
    batch_size=1, multiple=False, sequence_lengths=[32]
) -> EndpointConfig:
    return EndpointConfig(
        name="asdf",
        endpoint="qwer",
        task="question-answering",
        model="zxcv",
        batch_size=batch_size,
        accept_multiples_of_batch_size=multiple,
        bucketing=SequenceLengthsConfig(sequence_lengths=sequence_lengths),
    )


def test_endpoint_config_to_pipeline_copy_fields():
    cfg = _endpoint_config_to_pipeline_config(_model_helper(1, True))
    assert cfg.task == "question-answering"
    assert cfg.model_path == "zxcv"


def test_endpoint_config_to_pipeline_config_dynamic_batch():
    cfg = _endpoint_config_to_pipeline_config(_model_helper(1, True))
    assert cfg.batch_size is None


def test_endpoint_config_to_pipeline_config_static_batch():
    cfg = _endpoint_config_to_pipeline_config(_model_helper(1, False))
    assert cfg.batch_size == 1

    cfg = _endpoint_config_to_pipeline_config(_model_helper(64, False))
    assert cfg.batch_size == 64


def test_endpoint_config_to_pipeline_config_scheduler():
    cfg = _endpoint_config_to_pipeline_config(_model_helper(sequence_lengths=[32]))
    assert cfg.kwargs == {"sequence_length": 32}

    cfg = _endpoint_config_to_pipeline_config(_model_helper(sequence_lengths=[32, 64]))
    assert cfg.kwargs == {"sequence_length": [32, 64]}
