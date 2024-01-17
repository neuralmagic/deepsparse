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

from deepsparse.utils.time import TimerManager


class AddMeasurementsSetterTimerManager(TimerManager):
    def set_measurements(self, measurements: List[Dict[str, float]]):
        with self.lock:
            self.measurements = measurements


def test_timer_manager_average():
    """average() attribute check"""

    timer_manager = AddMeasurementsSetterTimerManager()

    timer_manager.set_measurements(
        [
            {"foo": [1, 2, 3], "bar": [1, 2], "giz": [1, 2, 3, 4]},
            {"foo": [1, 5], "bar": [1, 2], "baz": [3, 4]},
        ]
    )

    expected = {
        "time": {"foo": 2.4, "bar": 1.5, "baz": 3.5, "giz": 2.5},
        "iteration": {"foo": 2.5, "bar": 2, "baz": 2, "giz": 4},
    }
    assert timer_manager.average() == expected
