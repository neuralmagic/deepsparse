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

import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy
from pydantic import BaseModel

import pytest
from deepsparse import Pipeline


class FakeSchema(BaseModel):
    sentence: str


class FakePipeline(Pipeline):
    """
    A pipeline built specifically for testing the internals of pipeline.
    See process_inputs for more details.

    This does not actually initialize the engine.
    """

    def setup_onnx_file_path(self) -> str:
        return ""

    def _initialize_engine(self):
        return None

    def engine_forward(self, engine_inputs):
        # NOTE: this is overriden so we don't have to actually call into the engine
        return engine_inputs

    @property
    def input_schema(self) -> BaseModel:
        return FakeSchema

    @property
    def output_schema(self) -> BaseModel:
        return FakeSchema

    def process_inputs(self, inputs: FakeSchema):
        """
        Considers words to be a single batch:
        - the sentence "word" has a batch size of 1.
        - the sentence "the cat is orange" has a batch size of 4

        Each word gets transformed into 2 numpy arrays, so the list output
        of this function has 2 elements:
        1. A count of the characters in the word. Shape (256, )
        2. A count of the upper and lower case characters in the word. Shape (2, )

        This lets us test:
        1. multiple batch sizes easily
        2. multiply numpy arrays
        """
        words = inputs.sentence.split()
        character_counts = []
        case_counts = []
        for word in words:
            char_count = numpy.zeros((256,), dtype=float)
            case_count = numpy.zeros((2,), dtype=float)
            for c in word:
                char_count[ord(c.lower())] += 1.0
                case_count[int(c.isupper())] += 1.0
            character_counts.append(char_count)
            case_counts.append(case_count)
        return [numpy.stack(character_counts), numpy.stack(case_counts)], {
            "num_words": len(words)
        }

    def process_engine_outputs(self, engine_outputs, **kwargs) -> FakeSchema:
        """
        This function does a dumb reconstruction of the input words based on the counts
        """
        assert len(engine_outputs) == 2
        num_words = kwargs["num_words"]
        character_counts, case_counts = engine_outputs
        assert character_counts.shape == (num_words, 256)
        assert case_counts.shape == (num_words, 2)

        words = []
        for char_count, case_count in zip(character_counts, case_counts):
            word = "".join(chr(c) * int(char_count[c]) for c in range(256))
            if case_count[int(True)] > case_count[int(False)]:
                word = word.upper()
            words.append(word)
        return FakeSchema(sentence=" ".join(words))


def test_examples():
    pipeline = FakePipeline("")
    outputs = pipeline(sentence="hello MY nAME is bob")
    assert outputs.sentence == "ehllo MY AEMN is bbo"


def test_split_engine_inputs():
    pipeline = FakePipeline("")
    inp = [numpy.zeros((4, 28)), numpy.zeros((4, 28)), numpy.zeros((4, 28))]

    out = pipeline.split_engine_inputs(inp, batch_size=4)
    assert numpy.array(out).shape == (1, 3, 4, 28)

    out = pipeline.split_engine_inputs(inp, batch_size=2)
    assert numpy.array(out).shape == (2, 3, 2, 28)

    out = pipeline.split_engine_inputs(inp, batch_size=1)
    assert numpy.array(out).shape == (4, 3, 1, 28)


def test_join_opposite_of_split():
    pipeline = FakePipeline("")
    inp = [
        numpy.random.rand(4, 28),
        numpy.random.rand(4, 28),
        numpy.random.rand(4, 28),
    ]

    out = pipeline.split_engine_inputs(inp, batch_size=2)
    assert numpy.array(out).shape == (2, 3, 2, 28)

    joined = pipeline.join_engine_outputs(out)
    assert numpy.array(joined).shape == (3, 4, 28)

    for i, j in zip(inp, joined):
        assert (i == j).all()


def test_split_engine_inputs_uneven_raises_error():
    pipeline = FakePipeline("")
    with pytest.raises(
        RuntimeError,
        match="batch size of 3 passed into pipeline "
        "is not divisible by model batch size of 2",
    ):
        pipeline.split_engine_inputs([numpy.zeros((3, 28))], batch_size=2)


@mock.patch(
    "tests.deepsparse.pipelines.test_pipeline.FakePipeline.engine_forward",
    side_effect=lambda x: x,
)
def test_pipeline_split_batches_into_1(engine_forward: mock.Mock):
    pipeline = FakePipeline("", batch_size=1)
    pipeline(sentence="word")
    assert engine_forward.call_count == 1

    engine_forward.reset_mock()

    pipeline(sentence="two words")
    assert engine_forward.call_count == 2

    engine_forward.reset_mock()

    pipeline(sentence="two words for me")
    assert engine_forward.call_count == 4


@mock.patch(
    "tests.deepsparse.pipelines.test_pipeline.FakePipeline.engine_forward",
    side_effect=lambda x: x,
)
def test_pipeline_split_batches_into_2(engine_forward):
    pipeline = FakePipeline("", batch_size=2)

    with pytest.raises(RuntimeError, match="is not divisible"):
        pipeline(sentence="word")

    pipeline(sentence="two words")
    assert engine_forward.call_count == 1

    engine_forward.reset_mock()

    pipeline(sentence="two words for me")
    assert engine_forward.call_count == 2


def test_pipeline_executor_num_workers():
    pipeline = FakePipeline("", batch_size=2)
    assert pipeline.executor._max_workers == 1

    pipeline = FakePipeline("", batch_size=2, executor=2)
    assert pipeline.executor._max_workers == 2

    pipeline = FakePipeline("", batch_size=None, executor=2)
    assert pipeline.executor._max_workers == 2

    pipeline = FakePipeline("", batch_size=None, executor=ThreadPoolExecutor(3))
    assert pipeline.executor._max_workers == 3

    pipeline = FakePipeline("", batch_size=1, executor=ThreadPoolExecutor(3))
    assert pipeline.executor._max_workers == 3

    pipeline = FakePipeline("", batch_size=None)
    assert pipeline.executor._max_workers >= 1


def sum_sleep(xs):
    ms_to_sleep = xs[0].sum() * 10
    time.sleep(ms_to_sleep / 1000.0)
    return xs


@mock.patch(
    "tests.deepsparse.pipelines.test_pipeline.FakePipeline.engine_forward",
    side_effect=sum_sleep,
)
def test_pipeline_call_is_async(engine_forward):
    # here we make engine_forward take a different amount of time
    # based on the input. the total time should be the longest
    # forward call, instead of sum.
    #
    # The time each forward takes is just summing up character counts
    # in each word. So "aaaa" will sleep for 50ms whereas "a" will sleep for 10ms
    executor = ThreadPoolExecutor(max_workers=2)
    pipeline = FakePipeline("", batch_size=1, executor=executor)

    # sanity check that one input takes the correct amount of time
    start = time.time_ns()
    pipeline(sentence="a" * 5)
    end = time.time_ns()
    dur_ms = (end - start) * 1e-6
    assert abs(dur_ms - 50) < 5

    # two words, total time should be 60ms
    start = time.time_ns()
    pipeline(sentence="abcdef qwe")
    end = time.time_ns()
    dur_ms = (end - start) * 1e-6
    assert abs(dur_ms - 60) < 5

    # now check that with an executor with 1 worker takes 90s
    executor = ThreadPoolExecutor(max_workers=1)
    pipeline = FakePipeline("", batch_size=1, executor=executor)
    start = time.time_ns()
    pipeline(sentence="abcdef qwe")
    end = time.time_ns()
    dur_ms = (end - start) * 1e-6
    assert abs(dur_ms - 90) < 5
