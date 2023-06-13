#############################################################################
# This is a placeholder class to be replaced by LIB.deepsparse_engine.KVCache
#############################################################################

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

from typing import Dict, List

import numpy


class KVCache:
    def __init__(
        self,
        cache: Dict[str, numpy.ndarray],
        prompt_length: int,
        frozen_positions: List[int],
    ):
        pass

    def reset(self, prompt_length: int, frozen_positions: List[int]):
        pass

    def shift_last(count: int):
        pass
