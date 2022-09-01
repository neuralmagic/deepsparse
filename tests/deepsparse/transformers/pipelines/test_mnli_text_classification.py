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

from contextlib import suppress as do_not_raise

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
    (
        "batch_size,num_static_labels,num_sequences,num_labels,"
        "creation_expectation,inference_expectation"
    ),
    [
        (None, 1, 1, 1, do_not_raise(), pytest.raises(ValueError)),
        (None, 1, 1, None, do_not_raise(), do_not_raise()),
        (1, 1, 1, 1, do_not_raise(), pytest.raises(ValueError)),
        (1, 1, 1, None, do_not_raise(), do_not_raise()),
        (7, 13, 7, None, do_not_raise(), do_not_raise()),
        (13, None, 7, 13, do_not_raise(), do_not_raise()),
        (2, 13, 7, None, do_not_raise(), pytest.raises(ValueError)),
        (2, None, 7, 13, do_not_raise(), pytest.raises(ValueError)),
    ],
)
@pytest.mark.smoke
@mock_engine(rng_seed=0)
def test_zscp_pipeline(
    engine,
    batch_size,
    num_static_labels,
    num_sequences,
    num_labels,
    creation_expectation,
    inference_expectation,
    model_stub,
):
    static_labels = _generate_texts(num_static_labels)
    sequences = _generate_texts(num_sequences)
    labels = _generate_texts(num_labels)

    with creation_expectation:
        pipeline = Pipeline.create(
            "zero_shot_text_classification",
            model_path=model_stub,
            batch_size=batch_size,
            labels=static_labels,
        )

    with inference_expectation:
        pipeline(sequences=sequences, labels=labels)


def _generate_texts(num_texts):
    return ["sample_text"] * num_texts if num_texts else None
