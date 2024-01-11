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


from typing import Any, Callable, Dict, Optional

from .frequency_logger import FrequencyLogger


class ListLogger(FrequencyLogger):
    def __init__(self, frequency: int = 1, handler: Optional[Dict] = None):
        super().__init__(frequency)
        self.logs = []
        self.logger = lambda x: self.logs.append(x)

    def log(
        self,
        value: Any,
        tag: str,
        func: Optional[Callable] = None,
    ):
        super().log(logger=self.logger, value=value, tag=tag, func=func)
