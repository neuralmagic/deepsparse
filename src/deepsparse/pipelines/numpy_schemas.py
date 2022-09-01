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

from typing import Generic, Type, TypeVar

import numpy
from pydantic import ValidationError
from pydantic.fields import ModelField


Dtype = TypeVar("Dtype")


class NumpyArray(Generic[Dtype]):
    """
    `pydantic` compatible numpy.ndarray.

    Examples:
    ```python
    class Model(BaseModel):
        any_dtype: NumpyArray
        f64_array: NumpyArray[Float64]
        u8_array: NumpyArray[Uint8]
    ```

    See https://pydantic-docs.helpmanual.io/usage/types/#generic-classes-as-types
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: ModelField) -> numpy.ndarray:
        if not isinstance(v, numpy.ndarray):
            raise TypeError(f"Expected {numpy.ndarray}, found {v.__class__}")

        if not field.sub_fields:
            # Generic parameters were not provided, any dtype passes
            return v

        dtype_f: ModelField = field.sub_fields[0]
        _, error = dtype_f.validate(v.dtype, {}, loc="dtype")
        if error:
            raise ValidationError([error], cls)
        return v


class _NumpyDtypeValidator:
    expected: Type[numpy.dtype]

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v) -> numpy.ndarray:
        if not isinstance(v, numpy.dtype):
            raise TypeError(f"Expected {numpy.dtype}, found {v.__class__}")
        if numpy.issubdtype(v, cls.expected):
            raise ValueError(f"Expected dtype {cls.expected}, found {v}")
        return v


class Bool(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.bool8


class UInt8(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.uint8


class UInt16(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.uint16


class UInt32(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.uint32


class UInt64(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.uint64


class Float16(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.float16


class Float32(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.float32


class Float64(_NumpyDtypeValidator):
    expected: Type[numpy.dtype] = numpy.float64
