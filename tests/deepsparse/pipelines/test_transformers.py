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

from deepsparse import Pipeline
from tests.utils import mock_engine


@mock_engine(rng_seed=0)
def test_question_answering_pipeline(engine_mock):
    pipeline = Pipeline.create("question_answering", version_2_with_negative=True)
    output = pipeline(question="who am i", context="i am corey")
    assert output.answer == ""


@mock_engine(rng_seed=0)
def test_text_classification_pipeline(engine_mock):
    pipeline = Pipeline.create("text_classification")
    output = pipeline(["this is a good sentence", "this sentence is terrible!"])
    assert len(output.labels) == 2
    assert len(output.scores) == 2


@mock_engine(rng_seed=0)
def test_token_classification_pipeline(engine_mock):
    pipeline = Pipeline.create("token_classification")
    output = pipeline(
        [
            "My name is Bob and I live in Boston",
            "Sally Bob Joe. California, United States. Europe",
        ]
    )
    assert len(output.predictions) == 2
    assert len(output.predictions[0]) == 9
    assert len(output.predictions[1]) == 10

    assert [p.word for p in output.predictions[0]] == [
        "my",
        "name",
        "is",
        "bob",
        "and",
        "i",
        "live",
        "in",
        "boston",
    ]

    assert [p.word for p in output.predictions[1]] == [
        "sally",
        "bob",
        "joe",
        ".",
        "california",
        ",",
        "united",
        "states",
        ".",
        "europe",
    ]
