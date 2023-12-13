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

import time
from typing import Any, Dict

from deepsparse.middlewares.middleware import MiddlewareCallable


class TimerMiddleware(MiddlewareCallable):
    def __init__(self, call_next: MiddlewareCallable, identifier: str):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:
        start_time = time.time()
        result = self.call_next(*args, **kwargs)
        measurement = time.time() - start_time

        self.send(self.reducer, measurement)

        return result

    def reducer(self, state: Dict, *args, **kwargs):
        if "measurements" not in state:
            state["measurements"] = []
        state["measurements"].append(args[0])

        return state
