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

from contextlib import nullcontext

import numpy

import pytest
from deepsparse import Pipeline
from tests.utils import mock_engine


@pytest.fixture(scope="session")
def model_stub():
    return (
        "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/"
        "mnli/pruned80_quant-none-vnni"
    )


@pytest.mark.parametrize(
    "batch_size,num_sequences,num_static_labels,num_dynamic_labels,inference_error",
    [
        (1, 1, 1, 1, ValueError),  # both labels provided
        (1, 1, None, None, ValueError),  # no labels provided
        (1, 1, 1, None, None),  # static labels
        (1, 1, None, 1, None),  # dynamic labels
        # pass string sequence
        (1, "test_sequence", None, 1, None),
        (1, "test_sequence", 1, None, None),
        # batch_size 1
        (1, 13, 7, None, None),
        (1, 7, 13, None, None),
        (1, 13, None, 7, None),
        (1, 7, None, 13, None),
        # batch_size divides inputs
        (7, 13, 7, None, None),
        (13, 7, 13, None, None),
        (7, 13, None, 7, None),
        (13, 7, None, 13, None),
        # batch_size does not divide inputs
        (2, 13, 7, None, ValueError),
        (2, 7, 13, None, ValueError),
        (2, 13, None, 7, ValueError),
        (2, 7, None, 13, ValueError),
    ],
)
@pytest.mark.smoke
@mock_engine(rng_seed=0)
def test_batch_size(
    engine,
    batch_size,
    num_sequences,
    num_static_labels,
    num_dynamic_labels,
    inference_error,
    model_stub,
):
    static_labels = _generate_texts(num_static_labels)
    sequences = _generate_texts(num_sequences)
    dynamic_labels = _generate_texts(num_dynamic_labels)

    pipeline = Pipeline.create(
        "zero_shot_text_classification",
        model_path=model_stub,
        batch_size=batch_size,
        labels=static_labels,
    )

    with pytest.raises(inference_error) if inference_error else nullcontext():
        pipeline(sequences=sequences, labels=dynamic_labels)


@pytest.mark.parametrize(
    (
        "num_sequences,num_static_labels,num_dynamic_labels,"
        "exp_sequences_shape,exp_labels_shape,exp_scores_shape"
    ),
    [
        # static batch
        (3, 5, None, (3,), (3, 5), (3, 5)),
        (5, 3, None, (5,), (5, 3), (5, 3)),
        # dynamic batch
        (3, None, 5, (3,), (3, 5), (3, 5)),
        (5, None, 3, (5,), (5, 3), (5, 3)),
        # passing string removes the first dimension
        (1, 3, None, (1,), (1, 3), (1, 3)),
        ("test_sequence", 3, None, (), (3,), (3,)),
    ],
)
@pytest.mark.smoke
@mock_engine(rng_seed=0)
def test_output_shapes(
    engine,
    num_sequences,
    num_static_labels,
    num_dynamic_labels,
    exp_sequences_shape,
    exp_labels_shape,
    exp_scores_shape,
    model_stub,
):
    static_labels = _generate_texts(num_static_labels)
    sequences = _generate_texts(num_sequences)
    dynamic_labels = _generate_texts(num_dynamic_labels)

    pipeline = Pipeline.create(
        "zero_shot_text_classification",
        model_path=model_stub,
        labels=static_labels,
    )

    pipeline_output = pipeline(sequences=sequences, labels=dynamic_labels)
    assert numpy.array(pipeline_output.sequences).shape == exp_sequences_shape
    assert numpy.array(pipeline_output.labels).shape == exp_labels_shape
    assert numpy.array(pipeline_output.scores).shape == exp_scores_shape


def _generate_texts(num_texts):
    if isinstance(num_texts, int):
        return ["sample_text"] * num_texts
    else:
        return num_texts
