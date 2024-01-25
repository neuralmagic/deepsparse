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

from collections import defaultdict
from threading import Lock


class FrequencyFilter:
    def __init__(self):
        self._lock = Lock()
        self.counter = defaultdict(int)

    def inc(self, tag: str, func: str) -> None:
        """
        Increment the counter with respect to tag and func

        :param tag: Tag fro the config file
        :param func: Name of the func from the config file

        """
        stub = f"{tag}.{func}"
        with self._lock:
            self.counter[stub] += 1

    def should_execute_on_frequency(self, tag: str, func: str, freq: int) -> bool:
        """
        Check if the given tag, func and freq satisfies the criteria to execute.
        If the counter with respect to tag and func is a multiple of freq, then
        execute

        :param tag: Tag fro the config file
        :param func: Name of the func from the config file
        :param freq: The rate to log from the config file

        """

        stub = f"{tag}.{func}"
        with self._lock:
            counter = self.counter[stub]

        return counter % freq == 0
