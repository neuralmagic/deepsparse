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

import numpy

import pytest
from deepsparse import Pipeline
from tests.utils import mock_engine


@pytest.fixture(scope="session")
def model_stub():
    return (
        "zoo:nlp/token_classification/distilbert-none/pytorch/huggingface/"
        "conll2003/pruned80_quant-none-vnni"
    )


@pytest.mark.parametrize(
    "num_sequences,aggregation_strategy",
    [
        (1, "none"),
        (1, "simple"),
        (1, "first"),
        (1, "average"),
        (1, "max"),
    ],
)
@pytest.mark.smoke
@mock_engine(rng_seed=0)
def test_aggregation_strategy(
    engine,
    num_sequences,
    aggregation_strategy,
    model_stub,
):
    sequences = _generate_texts(num_sequences)

    pipeline = Pipeline.create(
        "token_classification",
        model_path=model_stub,
        batch_size=1,
        aggregation_strategy=aggregation_strategy,
    )

    print(sequences)
    out = pipeline(sequences)
    print(out)


def _generate_texts(num_texts):
    if isinstance(num_texts, int):
        return ["sample_text"] * num_texts
    else:
        return num_texts
