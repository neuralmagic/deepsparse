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

import yaml

import pytest
from deepsparse.server.config import (
    EndpointConfig,
    ImageSizesConfig,
    SequenceLengthsConfig,
    ServerConfig,
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

    cfg = EndpointConfig(task="", model="").to_pipeline_config()
    assert cfg.batch_size == 1

    cfg = EndpointConfig(task="", model="", batch_size=64).to_pipeline_config()
    assert cfg.batch_size == 64


def test_yaml_load_config(tmp_path):
    server_config = ServerConfig(
        num_cores=1,
        num_workers=2,
        integration="sagemaker",
        endpoints=[
            EndpointConfig(
                name="asdf",
                route="qwer",
                task="uiop",
                model="hjkl",
                batch_size=1,
                bucketing=None,
            ),
            EndpointConfig(
                name="asdf",
                route="qwer",
                task="uiop",
                model="hjkl",
                batch_size=2,
                bucketing=ImageSizesConfig(image_sizes=[(1, 1), (2, 2)]),
            ),
            EndpointConfig(
                name="asdf",
                route="qwer",
                task="uiop",
                model="hjkl",
                batch_size=3,
                bucketing=SequenceLengthsConfig(sequence_lengths=[5, 6, 7]),
            ),
        ],
        loggers="path/to/logging/config.yaml",
    )

    path = tmp_path / "config.yaml"
    with open(path, "w") as fp:
        yaml.dump(server_config.dict(), fp)

    with open(path) as fp:
        obj = yaml.load(fp, Loader=yaml.Loader)
    server_config2 = ServerConfig(**obj)
    assert server_config == server_config2
