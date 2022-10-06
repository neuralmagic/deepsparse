"""
Container how all the estimators available for the
data logging
"""
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

from typing import Optional, Tuple, Union

import numpy


__all__ = ["min_estimator"]


def min_estimator(
    input: numpy.ndarray, axis: Optional[Union[int, Tuple]] = None
) -> numpy.ndarray:
    if axis:
        return numpy.min(input, axis=axis)
    return numpy.min(input)


def max_estimator():
    raise NotImplementedError


def percentage_zeros_estimator():
    raise NotImplementedError


def mean_estimator():
    raise NotImplementedError


def std_estimator():
    raise NotImplementedError
