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

from typing import Any

from deepsparse.middlewares.middleware import MiddlewareCallable


IS_NESTED_KEY = "is_nested"
NAME_KEY = "name"
INFERENCE_STATE_KEY = "inference_state"


class TimerMiddleware(MiddlewareCallable):
    def __init__(
        self, call_next: MiddlewareCallable, identifier: str = "TimerMiddleware"
    ):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:
        name = kwargs.get(NAME_KEY)
        is_nested = kwargs.pop(IS_NESTED_KEY, False)

        inference_state = kwargs.get(INFERENCE_STATE_KEY)
        if inference_state and hasattr(inference_state, "timer"):
            timer = inference_state.timer
            with timer.time(id=name, enabled=not is_nested):
                return self.call_next(*args, **kwargs)
        return self.call_next(*args, **kwargs)
