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

import pytest
from deepsparse.utils import arrays_to_bytes, bytes_to_arrays


@pytest.mark.parametrize(
    "arrays",
    (
        [
            numpy.random.randint(255, size=(3, 244, 244), dtype=numpy.uint8)
            for _ in range(10)
        ],
        [numpy.random.randn(3, 224, 224).astype(numpy.float32) for _ in range(10)],
        [numpy.random.randn(i * 5, i * 20) for i in range(5)],
        [numpy.random.randint(255, size=(3, 244, 244), dtype=numpy.uint8)],
    ),
)
def test_arrays_bytes_conversion(arrays):
    serialized_arrays = arrays_to_bytes(arrays)
    assert isinstance(serialized_arrays, bytearray)

    deserialized_arrays = bytes_to_arrays(serialized_arrays)
    assert isinstance(deserialized_arrays, list)
    assert len(deserialized_arrays) == len(arrays)

    for array, deserialized_array in zip(arrays, deserialized_arrays):
        assert isinstance(deserialized_array, numpy.ndarray)
        assert array.shape == deserialized_array.shape
        assert numpy.all(array == deserialized_array)
