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

from typing import List

import numpy


def tensors_to_bytes(arrays: List[numpy.array]) -> str:
    to_return = bytearray()
    for arr in arrays:
        arr_dtype = bytearray(str(arr.dtype), "utf-8")
        arr_shape = bytearray(",".join([str(a) for a in arr.shape]), "utf-8")
        sep = bytearray("|", "utf-8")
        arr_bytes = arr.ravel().tobytes()
        to_return += arr_dtype + sep + arr_shape + sep + arr_bytes
    return to_return


def bytes_to_tensors(serialized_arr: str) -> List[numpy.array]:
    sep = "|".encode("utf-8")
    arrays = []
    i_start = 0
    while i_start < len(serialized_arr) - 1:
        i_0 = serialized_arr.find(sep, i_start)
        i_1 = serialized_arr.find(sep, i_0 + 1)
        arr_dtype = numpy.dtype(serialized_arr[i_start:i_0].decode("utf-8"))
        arr_shape = tuple(
            [int(a) for a in serialized_arr[i_0 + 1 : i_1].decode("utf-8").split(",")]
        )
        arr_num_bytes = numpy.prod(arr_shape) * arr_dtype.itemsize
        arr_str = serialized_arr[i_1 + 1 : arr_num_bytes + (i_1 + 1)]
        arr = numpy.frombuffer(arr_str, dtype=arr_dtype).reshape(arr_shape)
        arrays.append(arr.copy())

        i_start = i_1 + arr_num_bytes + 1
    return arrays
