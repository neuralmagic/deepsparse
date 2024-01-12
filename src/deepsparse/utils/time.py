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
from dataclasses import dataclass
from typing import Dict, List


__all__ = ["Timer", "InferenceStages", "TIMER_KEY"]

TIMER_KEY = "timer"


@dataclass(frozen=True)
class InferenceStages:
    TOTAL_INFERENCE: str = "total_inference"


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

    def __repr__(self):
        """
        Provide a string representation of the TimerManager object.

        :return: a string representing the timer manager object with its times.
        """
        return f"TimerManager({self.measurements})"

    def get_new_timer(self):
        return Timer()

    def update(self, measurements: Dict[str, float]):
        with self.lock:
            self.measurements.append(measurements)

    def average(self):
        """Get the average time for each key in every element inside measurements"""

        averages = {"time": {}, "iteration": {}}
        counts = {}

        for measurement in self.measurements:
            for key, values in measurement.items():
                averages["time"][key] = averages["time"].get(key, 0) + sum(values)
                averages["iteration"][key] = averages["iteration"].get(key, 0) + len(
                    values
                )
                counts[key] = counts.get(key, 0) + 1

        for key in averages["time"]:
            averages["time"][key] /= averages["iteration"][key]
            averages["iteration"][key] /= counts[key]

        return averages
