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

import uuid

import numpy


__all__ = ["softmax", "generate_session_id"]


def softmax(x: numpy.ndarray) -> numpy.ndarray:
    """
    Compute softmax values for x. This function is
    against overflow/underflow by using the
    trick of shifting the input vector by subtracting
    the maximum element in it from all elements

    :param x: input array
    :return: softmax values
    """
    z = x - max(x)
    numerator = numpy.exp(z)
    denominator = numpy.sum(numerator)
    return numerator / denominator


def generate_session_id() -> str:
    """
    Generate uuid for session id. This is used to
    identify the kv cache session for the user
    """
    session_id = str(uuid.uuid4())
    return session_id
