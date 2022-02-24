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

"""
Utilities for serving models in the DeepSparse server
"""

from typing import Any

import numpy


__all__ = ["serializable_response"]


def serializable_response(data: Any) -> Any:
    """
    :param data: input data to correct for serialization such as changing numpy
        arrays to primitives
    :return: a response that can be used for serialization such as through json
        for the serving
    """
    if not data:
        return data

    if isinstance(data, numpy.generic):
        return data.item()

    if isinstance(data, dict):
        for key in list(data.keys()):
            data[key] = serializable_response(data[key])
    elif isinstance(data, list):
        for index in range(len(data)):
            data[index] = serializable_response(data[index])

    return data
