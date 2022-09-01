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
from pydantic.fields import ModelField


Dtype = TypeVar("Dtype")


class NumpyArray(Generic[Dtype]):
    """
    `pydantic` compatible numpy.ndarray.

    Examples:
    ```python
    from deepsparse.pipelines.numpy_schemas import NumpyArray, UInt8, Float64
    class Model(BaseModel):
        any_dtype: NumpyArray
        any_sized_float: NumpyArray[float]
        f64: NumpyArray[Float64]
        u8: NumpyArray[UInt8]
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
        validator = dtype_f.type_
        if issubclass(validator, _NumpyDtypeValidator):
            expected_dtype = validator.expected
        else:
            possible_raw_types = {
                float: numpy.floating,
                int: numpy.integer,
                bool: numpy.bool8,
            }
            if dtype_f.type_ not in possible_raw_types:
                raise TypeError(
                    f"Invalid generic parameter of NumpyArray {dtype_f.type_}"
                )
            expected_dtype = possible_raw_types[dtype_f.type_]

        if not numpy.issubdtype(v.dtype, expected_dtype):
            raise TypeError(f"Expected dtype {expected_dtype}, found {v.dtype}")
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
