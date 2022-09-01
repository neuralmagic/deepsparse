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
from pydantic import BaseModel, ValidationError

import pytest
from deepsparse.pipelines.numpy_schemas import Float32, NumpyArray, UInt8


def test_untyped():
    class Model(BaseModel):
        arr: NumpyArray

    Model(arr=numpy.array([], dtype=numpy.uint8))
    Model(arr=numpy.array([], dtype=numpy.int8))
    Model(arr=numpy.array([], dtype=numpy.bool8))
    Model(arr=numpy.array([], dtype=numpy.float32))
    Model(arr=numpy.array([], dtype=numpy.float64))


def test_float_array():
    class Model(BaseModel):
        arr: NumpyArray[float]

    Model(arr=numpy.array([], dtype=numpy.float16))
    Model(arr=numpy.array([], dtype=numpy.float32))
    Model(arr=numpy.array([], dtype=numpy.float64))


def test_float32_array():
    class Model(BaseModel):
        arr: NumpyArray[Float32]

    Model(arr=numpy.array([], dtype=numpy.float32))
    for dtype in [numpy.float16, numpy.float64, numpy.bool8, numpy.int64]:
        with pytest.raises(ValidationError):
            Model(arr=numpy.array([], dtype=dtype))


def test_uint8_array():
    class Model(BaseModel):
        arr: NumpyArray[UInt8]

    Model(arr=numpy.array([], dtype=numpy.uint8))

    for dtype in [
        numpy.float16,
        numpy.float32,
        numpy.float64,
        numpy.bool8,
        numpy.int64,
    ]:
        with pytest.raises(ValidationError):
            Model(arr=numpy.array([], dtype=dtype))


def test_invalid_raw_type():
    class Model(BaseModel):
        arr: NumpyArray[str]

    with pytest.raises(ValidationError, match="Invalid generic parameter"):
        Model(arr=numpy.array([], dtype=str))
