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

from typing import Any, List, Iterable

import numpy


def identity(x: Any):
    return x


def max(lst: Any):
    
    if isinstance(lst, Iterable) and len(lst) > 0:
        arr = numpy.array(lst)
        while arr.size > 1:
            arr = numpy.mean(arr)
        return arr.item()
    return lst


def average(lst: List):
    if len(lst) > 0:
        arr = numpy.array(lst)
        while arr.size > 1:
            arr = numpy.mean(arr)
        return arr.item()
