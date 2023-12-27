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
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Union

from deepsparse.utils.time import Timer


__all__ = ["State", "PipelineState", "InferenceState"]


class State(ABC):
    """
    Abstract class to store pipeline-level and inference-level state variables which
    are generated by some Operator, and required by some other Operator.
    """

    def __init__(self):
        super().__init__()
        self._current_state = None

    @property
    def current_state(self):
        return self._current_state


class PipelineState(State):
    """
    Created during pipeline initialization. Pipeline state values are ready-only
    duirng inference.
    """

    def create_state(self, new_state: dict):
        if self._current_state:
            raise ValueError("State creation is only allowed during initialization.")
        self._current_state = new_state


class TimerState:
    """TimerState shared among all InferenceState"""

    def __init__(self):
        super().__init__()
        self._timer = None

    @contextmanager
    def time(self, id: str):
        if self._timer is not None:
            with self.timer.time(id=id):
                yield
        else:
            yield  # null context

    def set_timer(self, timer: Timer):
        self._timer = timer

    @property
    def timer(self):
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):
        self._timer = timer


class InferenceState(State, TimerState):
    """
    Inference state, created during every inference run.
    """

    def __init__(self):
        super().__init__()

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

    def get_state(self, key: str):
        """Get value in current_state, if any"""
        if key in self.current_state:
            return self.current_state[key]

    def copy_state(self, props=["timer"]):
        """copy everything except the attrs in props"""

        original_values = {
            prop: getattr(self, prop) for prop in props if hasattr(self, prop)
        }
        for prop in props:
            setattr(self, prop, None)

        copied_state = deepcopy(self)

        for prop, value in original_values.items():
            setattr(copied_state, prop, value)
            setattr(self, prop, value)

        return copied_state
