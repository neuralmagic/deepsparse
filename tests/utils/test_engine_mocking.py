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
import re
from unittest.mock import MagicMock

import numpy

import pytest
from deepsparse import Context, Pipeline
from tests.utils.engine_mocking import SampleMode, _FakeDeepsparseLibEngine, mock_engine


@mock_engine(rng_seed=0)
def test_mock_engine_fakes_engine(engine_mock):
    pipeline = Pipeline.create("image_classification")
    assert isinstance(pipeline.engine._eng_net, _FakeDeepsparseLibEngine)


@mock_engine(rng_seed=0)
def test_mock_engine_calls(engine_mock: MagicMock):
    engine_mock.assert_not_called()
    # with pytest.raises(ValueError):
    context = Context(num_cores=1, num_streams=1)
    Pipeline.create("image_classification", batch_size=3, context=context)
    engine_mock.assert_called_once_with(
        os.path.join(
            os.path.expanduser("~"),
            ".cache/sparsezoo/84774c96-ab7d-4b3b-ab8c-2509d7bfcb09/model.onnx",
        ),
        3,
        1,
        1,
        "elastic",
        context.value,
    )


@mock_engine(rng_seed=0)
def test_mock_engine_checks_inputs_list(engine_mock: MagicMock):
    pipeline = Pipeline.create("image_classification")
    with pytest.raises(
        ValueError,
        match=re.escape("inp must be a list, given <class 'numpy.ndarray'>"),
    ):
        pipeline.engine(numpy.zeros((1, 3, 550, 550)))


@mock_engine(rng_seed=0)
def test_mock_engine_checks_inputs_dtype(engine_mock: MagicMock):
    pipeline = Pipeline.create("image_classification")
    with pytest.raises(
        AssertionError,
        match=re.escape("Expected <class 'numpy.float32'> found dtype=float64"),
    ):
        pipeline.engine([numpy.zeros((1, 3, 224, 224), dtype=numpy.float64)])


@mock_engine(rng_seed=0)
def test_mock_engine_checks_inputs_shape(engine_mock: MagicMock):
    pipeline = Pipeline.create("image_classification")
    with pytest.raises(
        AssertionError,
        match=re.escape("Expected [1, 3, 224, 224] found shape=(1, 3, 550, 550)"),
    ):
        pipeline.engine([numpy.zeros((1, 3, 550, 550))])


@mock_engine(rng_seed=0)
def test_mock_engine_outputs_rand(engine_mock: MagicMock):
    pipeline = Pipeline.create("image_classification")
    outputs = pipeline.engine([numpy.zeros((1, 3, 224, 224), dtype=numpy.float32)])
    assert len(outputs) == 2
    rng = numpy.random.default_rng(0)
    for out in outputs:
        assert numpy.all(out == rng.random(size=out.shape).astype(out.dtype))


@mock_engine(rng_seed=0, mode=SampleMode.zeros)
def test_mock_engine_outputs_zeros(engine_mock: MagicMock):
    pipeline = Pipeline.create("image_classification")
    outputs = pipeline.engine([numpy.zeros((1, 3, 224, 224), dtype=numpy.float32)])
    assert len(outputs) == 2
    for out in outputs:
        assert numpy.all(out == numpy.zeros(out.shape, dtype=out.dtype))
