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
import warnings
from abc import ABC
from typing import Any, Union


__all__ = ["State", "PipelineState", "InferenceState"]


class State(ABC):
    def __init__(self):
        self._current_state = None

    @property
    def current_state(self):
        return self._current_state


# Created during pipeline initialization; only read access
class PipelineState(State):
    def create_state(self, new_state: dict):
        if self._current_state:
            raise ValueError("State creation is only allowed during initialization.")
        self._current_state = new_state


# Should be created during each inference run, similar to the context
class InferenceState(State):
    def create_state(self, new_state: dict):
        if self._current_state:
            warnings.warn("Current state already exists, overriding.")
        self._current_state = new_state

    def update_value(self, attribute: str, value: Union[str, int, list]):
        if not self._current_state.get(attribute):
            raise ValueError(f"{attribute} is not a valid state attribute")
        self._current_state[attribute] = value

    def update_state(self, value: Any):
        self._current_state.update(value)
