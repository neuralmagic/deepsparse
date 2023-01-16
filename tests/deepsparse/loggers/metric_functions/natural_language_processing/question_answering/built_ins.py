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
from deepsparse.loggers.metric_functions import answer_length, answer_score
from deepsparse.transformers.pipelines.question_answering import QuestionAnsweringOutput


output_schema = QuestionAnsweringOutput(
    answer="His palms are sweaty", score=0.69, start=0, end=0
)


@pytest.mark.parametrize(
    "schema, expected_len",
    [
        (output_schema, 20),
    ],
)
def test_answer_length(schema, expected_len):
    assert answer_length(schema) == expected_len


@pytest.mark.parametrize(
    "schema, expected_score",
    [
        (output_schema, 0.69),
    ],
)
def test_answer_score(schema, expected_score):
    assert answer_score(schema) == expected_score
