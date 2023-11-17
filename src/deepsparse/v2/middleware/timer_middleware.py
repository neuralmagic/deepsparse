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
from functools import wraps
from typing import Optional

from deepsparse.v2.utils.state import InferenceState


class Timer:

    _lock = threading.Lock()
    condition = threading.Condition()

    def __init__(self):

        self.measurements = {}
        self.start_times = {}

    def start(self, key):
        with self._lock:
            self.start_times[key] = time.time()

    def end(self, key):
        end_time = time.time()

        # TODO: while waiting for N seconds <- condition
        if key in self.start_times:
            start_time = self.start_times[key]
            with self._lock:
                del self.start_times[key]
            self.measurements[key] = end_time - start_time
        return None

    def _timeout(self, wait: int = 0.1):
        time.wait(wait)
        return True

    def clock(timer, func):
        @wraps(func)
        def wrapper(self, inp, context):
            start_time = time.time()

            # Call the original method
            result = func(self, inp, context)

            end_time = time.time()
            runtime = end_time - start_time

            # Store the runtime with class name and function name as the key
            key = f"{self.__class__.__name__}.{func.__name__}"
            timer.measurements[key] = runtime

            return result

        return wrapper


class TimerMiddleware:

    timer = Timer()

    def start_event(self, name: str, state: Optional[InferenceState] = None):

        is_inference_run = state is not None
        is_state_coldstart = is_inference_run and not hasattr(state, "timer")
        is_new_name = (
            hasattr(state, "timer") and name not in getattr(state, "timer").start_times
        )

        if is_state_coldstart or is_new_name:
            # timer only exists in the context of a single inference pass
            # this way subtimings are correlated with the other timings in its pass
            state.timer = Timer()
            state.timer.start(name)
            return

        self.timer.start(name)

    def end_event(self, name, state: Optional[InferenceState] = None):
        """
        Populate the timer state. If state is provided, populate all its
        measurements into the timer state
        """
        if state and hasattr(state, "timer"):
            state.timer.end(name)

            if name in self.timer.measurements:
                print(f"warning, {name} already exists")
            with self.timer._lock:
                # self.timer.measurements[name] = state.timer.measurements[name]
                self.timer.measurements.update(state.timer.measurements)

        self.timer.end(name)
