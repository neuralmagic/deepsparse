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

from collections import defaultdict
from typing import List, Tuple, Union

import numpy

import pytest
from deepsparse.transformers.utils.token_generator import TokenGenerator


_MIN_FLOAT = numpy.finfo(numpy.float32).min


@pytest.fixture(scope="function")
def logits_fixture() -> numpy.array:
    def get(shape: Tuple = (1, 1, 51200), token_max_thresh: int = 30, low: int = -30):
        return numpy.random.uniform(low, token_max_thresh, size=shape)

    return get


@pytest.fixture(scope="function")
def token_fixture() -> List[int]:
    def get(shape: Union[int, Tuple] = 5, token_max_thresh: int = 51200):
        return numpy.random.randint(0, token_max_thresh, size=shape).tolist()

    return get


class TestTokenGenerator:
    def test_update_frequencies(
        self, logits_fixture, token_fixture, token_max_thresh: int = 51200
    ):
        logits, tokens = logits_fixture(), token_fixture(
            token_max_thresh=token_max_thresh
        )
        token_generator = TokenGenerator(
            logits_shape=logits[-1].shape[-1], tokens=tokens.copy()
        )

        assert token_generator.tokens == tokens

        freq = defaultdict(int)
        for token in token_generator.tokens:
            freq[token] += 1

        for key, value in freq.items():
            assert token_generator.token_frequencies[key] == value

        # test TokenGenerator._update_frequencies
        new_token = token_fixture(shape=1)[0]
        token_generator.tokens.append(new_token)
        token_generator._update_frequencies(new_token)

        assert token_generator.tokens == tokens + [new_token]
        freq[new_token] += 1
        for key, value in freq.items():
            assert token_generator.token_frequencies[key] == value

    def test_apply_frequency_penalty(
        self,
        logits_fixture,
        token_fixture,
    ):
        logits, tokens = logits_fixture(), token_fixture()
        frequency_penalty = 1.0
        token_generator = TokenGenerator(
            logits_shape=logits[-1].shape[-1],
            tokens=(tokens + tokens),
            frequency_penalty=frequency_penalty,
        )

        test_logits = token_generator.token_frequencies
        # numpy arrays by default are pass by ref
        new_logits = token_generator.apply_frequency_penalty(test_logits.copy())
        assert new_logits.shape == test_logits.shape
        assert numpy.sum(new_logits) == 0

    def test_apply_presence_penalty(
        self,
        logits_fixture,
        token_fixture,
    ):
        logits, tokens = logits_fixture(), token_fixture()
        presence_penalty = 1.0
        token_generator = TokenGenerator(
            logits_shape=logits[-1].shape[-1],
            tokens=(tokens + tokens),
            presence_penalty=presence_penalty,
        )
        test_logits = token_generator.token_frequencies
        # numpy arrays by default are pass by ref
        new_logits = token_generator.apply_presence_penalty(test_logits.copy())
        assert new_logits.shape == test_logits.shape
        assert numpy.sum(new_logits) == 0.5 * numpy.sum(test_logits)

    def test_apply_topk(
        self,
    ):
        # logits for opt usually have shape (1,1,51200)
        logits = numpy.linspace(0, 1, 11).reshape((1, 1, 11))

        token_generator = TokenGenerator(
            logits_shape=logits[-1].shape[-1],
            top_k=3,
        )

        filter_value = -float("Inf")
        new_logits = token_generator.apply_top_k(
            logits.copy(), filter_value=filter_value
        )

        for _ in range(token_generator.top_k):
            curr_max, idx = numpy.max(new_logits), numpy.argmax(new_logits)
            assert curr_max > filter_value
            new_logits = numpy.delete(new_logits, idx)

        assert numpy.all(new_logits == filter_value)

    @pytest.mark.parametrize(
        ("logits", "top_p", "min_tokens_to_keep", "expected_filtered_values"),
        [
            (
                0.1 * numpy.ones(10).reshape((1, 1, 10)),
                0.79,
                0,
                2,
            ),
            (
                0.1 * numpy.ones(10).reshape((1, 1, 10)),
                0.899,
                0,
                1,  # one token should have cumsum > 0.9
            ),
            (0.1 * numpy.ones(10).reshape((1, 1, 10)), 0, 1, 9),  # keep all toks but 1
            (
                numpy.array([1.0, -3.1, 2.0, 3.1, -1.0, -2.0, 1.2, -1.2]).reshape(
                    1, 1, -1
                ),
                # expected distribution:
                # [0.0012, 0.0049, 0.0132, 0.023, 0.097, 0.188, 0.3914, 1]
                0.9,
                0,
                5,
            ),
        ],
    )
    def test_apply_top_p(
        self,
        logits,
        top_p,
        min_tokens_to_keep,
        expected_filtered_values,
    ):

        token_generator = TokenGenerator(
            logits_shape=logits[-1].shape[-1],
            top_p=top_p,
        )

        filter_value = _MIN_FLOAT
        new_logits = token_generator.apply_top_p(
            logits.copy(),
            filter_value=filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
        )
        assert (new_logits[-1] == filter_value).sum(axis=1) == expected_filtered_values

    def test_generate_token(
        self,
        logits_fixture,
        token_fixture,
    ):
        logits, tokens = logits_fixture(), token_fixture()
        token_generator = TokenGenerator(
            logits_shape=logits[-1].shape[-1],
            tokens=(tokens + tokens),
            deterministic=False,
        )
        new_token = token_generator.generate(logits=logits[0, -1, :])
        assert new_token == token_generator.tokens[-1]
        assert len(token_generator.tokens) == len(tokens + tokens) + 1
