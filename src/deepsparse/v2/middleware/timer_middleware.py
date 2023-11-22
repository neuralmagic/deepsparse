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
from typing import Dict, Optional

from deepsparse.v2.utils.state import InferenceState


class Timer:
    def __init__(self):
        self._lock = threading.Lock()
        self.measurements = {}
        self.start_times = {}

    def start(self, key: str):
        """
        Starts the timer
        :param key: the key to track the timer for
        """
        with self._lock:
            self.start_times[key] = time.time()

    def end(self, key: str) -> bool:
        """
        Ends the timer and saves the runtime into measurements
        :param key: the key to save the run time for
        """
        end_time = time.time()

        with self._lock:
            if key in self.start_times:
                start_time = self.start_times[key]
                del self.start_times[key]
                self.measurements[key] = end_time - start_time
                return True
        return False

    def update_measurements(self, key: str, measurements: Dict[str, float]):
        """
        Update the measurements from another timer. Used for
         saving inference state measurements into the middleware timer
        :param key: the key to save the run time for
        :param measurements: dict of measurements
        """
        with self._lock:
            self.measurements[key] = measurements
        return True


class TimerMiddleware:
    """
    Timer middleare to keep run times of itsself and inference state timer
    """

    def __init__(self):
        """
        Initialize timer for middleware state.
        """
        self.timer = Timer()

    def start_event(self, name: str, state: Optional[InferenceState] = None) -> None:
        """
        Start the timer for one of middleware state or inference state.
        Inference state timer will be used/initialized if state is provided
        """

        # middleware level timer
        if state is None:
            self.timer.start(name)
            return

        # state level timer
        if not hasattr(state, "timer"):
            state.timer = Timer()

        state.timer.start(name)

    def end_event(self, name: str, state: Optional[InferenceState] = None) -> bool:
        """
        End the timer for one of middleware state or inference state.
        Inference state timer will be used if state is provided
        """
        if state is None:
            self.timer.end(name)
            return True

        if hasattr(state, "timer"):
            state.timer.end(name)
            return True
        return False

    def update_middleware_timer(self, name: str, state: InferenceState) -> bool:
        """
        Write the inference state timer into the middleware state timer
        """
        if hasattr(state, "timer"):
            self.timer.update_measurements(name, state.timer.measurements)
            return True
        return False
