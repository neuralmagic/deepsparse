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
import numpy as np
from pydantic import BaseModel

import pytest
from deepsparse.utils import (
    join_engine_outputs,
    prep_for_serialization,
    split_engine_inputs,
)
from tests.server.test_helpers import DummyOutputSchema


def test_split_engine_inputs():
    inp = [numpy.zeros((4, 28)) for _ in range(3)]

    out, _ = split_engine_inputs(inp, batch_size=4)
    assert numpy.array(out).shape == (1, 3, 4, 28)

    out, _ = split_engine_inputs(inp, batch_size=2)
    assert numpy.array(out).shape == (2, 3, 2, 28)

    out, _ = split_engine_inputs(inp, batch_size=1)
    assert numpy.array(out).shape == (4, 3, 1, 28)


def test_join_opposite_of_split():
    inp = [numpy.random.rand(4, 28) for _ in range(3)]

    out, orig_batch_size = split_engine_inputs(inp, batch_size=2)
    assert numpy.array(out).shape == (2, 3, 2, 28)

    joined = join_engine_outputs(out, orig_batch_size)
    assert numpy.array(joined).shape == (3, 4, 28)

    for i, j in zip(inp, joined):
        assert (i == j).all()


def test_split_engine_inputs_uneven_pads():
    inp = [numpy.random.rand(3, 28)]

    out, orig_batch_size = split_engine_inputs(inp, batch_size=2)
    assert numpy.array(out).shape == (2, 1, 2, 28)

    joined = join_engine_outputs(out, orig_batch_size)
    assert numpy.array(joined).shape == (1, 3, 28)

    for i, j in zip(inp, joined):
        assert (i == j).all()


@pytest.mark.parametrize(
    "unserialized_output, target_serialized_output",
    [
        (
            DummyOutputSchema(
                field_1=[np.array([[1, 2, 3]])],
                field_2={"key_1": np.array([[[1, 2, 3]]])},
                field_3=DummyOutputSchema(field_1=np.array([0])),
            ),
            DummyOutputSchema(
                field_1=[[[1, 2, 3]]],
                field_2={"key_1": [[[1, 2, 3]]]},
                field_3=DummyOutputSchema(field_1=[0]),
            ),
        )
    ],
)
def test_prep_for_serialization(unserialized_output, target_serialized_output):
    def check_dict_equality(dict_1, dict_2):
        for key, value in dict_1.items():
            if isinstance(value, BaseModel):
                value = value.dict()
                check_dict_equality(value, dict_2[key].dict())
            elif isinstance(value, dict):
                check_dict_equality(value, dict_2[key])
            elif isinstance(value, list):
                equal = value == dict_2[key]
                equal = equal if isinstance(equal, bool) else equal.all()
                assert equal
            else:
                assert value == dict_2[key]

    serialized_output = prep_for_serialization(unserialized_output)
    serialized_output = serialized_output.dict()
    target_serialized_output = target_serialized_output.dict()
    check_dict_equality(target_serialized_output, serialized_output)
