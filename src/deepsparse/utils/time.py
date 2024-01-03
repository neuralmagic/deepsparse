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
    def time(self, id: str, enabled=True):
        if enabled:
            start = time.time()
            yield
            with self._lock:
                self.measurements[id].append(time.time() - start)
        else:
            yield


class TimerManager:
    """
    Timer to keep track of run times
    Multiple pipeline can use the same TimerManager instance
    Lock used for sharing measurements in all Operator per Pipeline
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.measurements: List[Dict] = []

    def get_new_timer(self):
        return Timer()

    def update(self, measurements: Dict[str, float]):
        with self.lock:
            self.measurements.append(measurements)

    def average(self):
        """Get the average time for each key in every element inside measurements"""

        summary = {"time": defaultdict(float), "execution": defaultdict(int)}

        for measurement in self.measurements:
            for key, values in measurement.items():
                summary["time"][key] += sum(values)
                summary["execution"][key] += len(values)

        for key in summary["time"]:
            summary["time"][key] /= summary["execution"][key]

        return {metric: dict(values) for metric, values in summary.items()}
