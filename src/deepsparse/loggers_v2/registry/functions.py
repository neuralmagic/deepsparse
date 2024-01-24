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

from typing import Any, Callable, Iterable

import numpy


__all__ = [
    "average",
    "identity",
    "max",
]


def identity(value: Any):
    return value


def max(lst: Any):
    return _apply_function_to_iterable(lst, numpy.max)


def average(lst: Any):
    return _apply_function_to_iterable(lst, numpy.mean)


def _apply_function_to_iterable(iterable: Iterable, func: Callable) -> Any:
    """
    Apply a callable to apply to an iterable. Used for dimentionality reduction
    to output a scalar

    :param iterable: An iterable
    :param func: the functiont to apply to the iterable to return a scalae

    Example:
        # Apply numpy.mean to an iterable
         _apply_function_to_iterable(iterable, numpy.mean)

    """
    if isinstance(iterable, Iterable) and len(iterable) > 0:
        if not isinstance(iterable, numpy.ndarray):
            iterable = numpy.array(iterable)

        if numpy.can_cast(iterable.dtype, numpy.number):
            arr = func(iterable)
            return arr.item()

    return iterable
