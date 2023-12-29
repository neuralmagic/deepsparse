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


import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List


class Timer:
    def __init__(self):
        self._lock = (
            threading.RLock()
        )  # reenterant lock, thread can acquire lock multiple times
        self.measurements = defaultdict(list)

    @contextmanager
    def time(self, id: str):
        start = time.time()
        yield
        with self._lock:
            self.measurements[id].append(time.time() - start)


class TimerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.measurements: List[Dict] = []

    def get_new_timer(self):
        return Timer()

    def update(self, measurements: Dict[str, float]):
        with self.lock:
            self.measurements.append(measurements)
