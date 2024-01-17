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
from typing import Any, Callable

from .pattern import is_match_found


class FrequencyFilter:
    def __init__(self):
        self._lock = Lock()
        self.frequency = {}
        self.counter = {}

    def add_template_to_frequency(self, tag: str, func: str, rate: int = 1):
        with self._lock:
            self.frequency[f".*{tag}.*{func}.*"] = rate

    def should_execute_on_frequency(
        self, tag: str, log_type: str, func: Callable
    ) -> bool:

        should_execute = False
        stub = f"{log_type}.{tag}.{func}"
        stub_frequency = f"{tag}.{func}"
        with self._lock:
            if stub not in self.counter:
                self.counter[stub] = 0

            for key, value in self.frequency.items():

                if is_match_found(key, stub_frequency):
                    frequency = value
                    self.counter[stub] = (self.counter[stub] + 1) % frequency
                    should_execute = should_execute or (self.counter[stub] == 0)

        return should_execute
