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

from .base_middleware import BaseMiddleware


class TimerMiddleware(BaseMiddleware):
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = {}
        self.measurements = {}

    def start_event(
        self,
        name: str,
        inputs: Optional[Dict] = None,
        inference_state: Optional["InferenceState"] = None,
        **kwargs,
    ):
        with self._lock:
            self.start_time[name] = time.time()

    def end_event(
        self,
        name: str,
        inputs: Optional[Dict] = None,
        inference_state: Optional["InferenceState"] = None,
        **kwargs,
    ):
        end_time = time.time()
        with self._lock:
            if name in self.start_time:
                start_time = self.start_time[name]
                del self.start_time[name]
                new_time = end_time - start_time
                if name in self.measurements:
                    self.measurements[name].append(new_time)
                else:
                    self.measurements[name] = [new_time]
