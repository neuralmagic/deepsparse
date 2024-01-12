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


from threading import Lock
from typing import Any, Callable, List, Optional


class FrequencyLogger:
    def __init__(
        self,
        frequency: Optional[int] = None,
    ):
        self._lock = Lock()
        self.frequency = frequency or 1
        self.counter = {}

    def log(
        self,
        logger: Callable,
        value: Any,
        tag: str,
        func: Optional[Callable] = None,
    ):
        stub = f"{tag}.{func.__name__}"
        if self.is_called_multiple_of_frequency(stub):
            if func is not None:
                value = func(value)
            logger(f"{value}, {tag}")

    def is_called_multiple_of_frequency(self, tag: str) -> bool:
        with self._lock:
            if tag not in self.counter:
                self.counter[tag] = 0
            self.counter[tag] += 1
            counter = self.counter.get(tag)

        if counter % self.frequency == 0:
            return True
        return False
