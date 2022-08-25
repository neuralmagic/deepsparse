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
import sys
from unittest.mock import patch

import pytest


SRC_DIRS = [
    os.path.join(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../examples"),
        dirname,
    )
    for dirname in [
        "twitter-nlp",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import analyze_sentiment
    import analyze_tokens


@pytest.mark.parametrize(
    "model, batch_size",
    (
        [
            pytest.param(
                m,
                b,
            )
            for m in [
                "zoo:nlp/token_classification/distilbert-none/pytorch/huggingface/conll2003/pruned80_quant-none-vnni"
            ]
            for b in [1, 16]
        ]
    ),
)
def test_analyze_tokens(model: str, batch_size: int):
    tweets_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../test_data/pineapple.txt"
    )
    testargs = f"""
        analyze_tokens.py
        --model_path {model}
        --batch_size {batch_size}
        --tweets_file {tweets_file}
        """.split()

    with patch.object(sys, "argv", testargs):
        analyze_tokens.analyze_tweets_tokens()


@pytest.mark.parametrize(
    "model, batch_size",
    (
        [
            pytest.param(
                m,
                b,
            )
            for m in [
                "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned80_quant-none-vnni"
            ]
            for b in [1, 16]
        ]
    ),
)
def test_analyze_sentiment(model: str, batch_size: int):
    tweets_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../test_data/pineapple.txt"
    )
    testargs = f"""
        analyze_sentiment.py
        --model_path {model}
        --batch_size {batch_size}
        --tweets_file {tweets_file}
        """.split()

    with patch.object(sys, "argv", testargs):
        analyze_sentiment.analyze_tweets_sentiment()
