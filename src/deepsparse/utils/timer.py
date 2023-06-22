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
from dataclasses import dataclass
from typing import Dict, List, Optional
import contextvars
from contextlib import contextmanager


__all__ = ["timer_context", "InferenceStages", "StagedTimer", "TimerManager"]


timer_context = contextvars.ContextVar("timer_context")


@dataclass(frozen=True)
class InferenceStages:
    PRE_PROCESS: str = "pre_process"
    ENGINE_FORWARD: str = "engine_forward"
    POST_PROCESS: str = "post_process"
    TOTAL_INFERENCE: str = "total_inference"


class StagedTimer:
    """
    Timer object that enables simultaneous starting and stopping of various stages.

    example usage of measuring two operation times with the overall time:

    ```python
    timer = StagedTimer()

    timer.start("overall_time")

    timer.start("operation_1")
    # DO OPERATION 1
    timer.stop("operation_1")

    timer.start("operation_2")
    # DO OPERATION 2
    timer.stop("operation_2")

    timer.stop("overall_time")
    ```

    :param enabled: if False, start/stop become no-ops. default True
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._staged_start_times = {}
        self._staged_stop_times = {}

    def __repr__(self):
        return f"StagedTimer({self.times})"

    @property
    def stages(self) -> List[str]:
        return list(self._staged_start_times.keys())

    @property
    def times(self) -> Dict[str, float]:
        return {stage: self.stage_average_time(stage) for stage in self.stages}

    @property
    def all_times(self) -> Dict[str, List[float]]:
        return {stage: self.stage_times(stage) for stage in self.stages}

    def clear(self):
        self._staged_start_times.clear()
        self._staged_stop_times.clear()

    def has_stage(self, stage: str) -> bool:
        return stage in self.stages

    def start(self, stage: str):
        if not self.enabled:
            return
        if stage not in self._staged_start_times:
            self._staged_start_times[stage] = []
            self._staged_stop_times[stage] = []

        if len(self._staged_start_times[stage]) != len(self._staged_stop_times[stage]):
            raise ValueError(
                f"Attempting to start {stage} before a previous has been stopped:"
                f" start times len({self._staged_start_times[stage]});"
                f" stop times len({self._staged_stop_times[stage]})"
            )

        self._staged_start_times[stage].append(time.perf_counter())

    def stop(self, stage: str):
        if not self.enabled:
            return
        if stage not in self._staged_start_times:
            raise ValueError(
                "Attempting to stop a stage that has not been started: " f"{stage}"
            )

        if (
            len(self._staged_start_times[stage])
            != len(self._staged_stop_times[stage]) + 1
        ):
            raise ValueError(
                f"Attempting to stop {stage} before a previous has been started:"
                f" start times len({self._staged_start_times[stage]});"
                f" stop times len({self._staged_stop_times[stage]})"
            )

        self._staged_stop_times[stage].append(time.perf_counter())

    def stage_times(self, stage: str) -> List[float]:
        if stage not in self._staged_start_times:
            raise ValueError(
                "Attempting to get time deltas for a stage that has not been started: "
                f"{stage}"
            )

        if len(self._staged_start_times[stage]) != len(self._staged_stop_times[stage]):
            raise ValueError(
                "Attempting to get time deltas for a stage that has not been stopped: "
                f"{stage}"
            )

        return [
            self._staged_stop_times[stage][i] - self._staged_start_times[stage][i]
            for i in range(len(self._staged_start_times[stage]))
        ]

    def stage_average_time(self, stage: str) -> float:
        times = self.stage_times(stage)

        return sum(times) / len(times)


class TimerManager:
    """
    Object to manage creation and aggregation of StagedTimers for benchmarking
    performance timings.

    Intended workflow:

    ```python
    timer_manager = TimerManager(multi=True)

    # process 1
    timer_1 = timer_manager.new_timer()
    timer_2.start(...)
    ...

    # process 2
    timer_2 = timer_manager.new_timer()
    timer_2.start(...)
    ...

    # aggregate times for benchmarking
    do_some_postprocessing(timer_manager.all_times)
    ```

    :param enabled: if False, no timings are measured by new staged timers. Default True
    :param multi: if True, keep track of all newly created staged timers. if False, only
        stores the latest created staged timer. Default False
    """

    def __init__(self, enabled: bool = True, multi: bool = False):
        self.multi = multi
        self.enabled = enabled
        self._timers = []

    def __repr__(self):
        return f"TimerManager({self.times})"

    @property
    def latest(self) -> Optional[StagedTimer]:
        return self._timers[-1] if self._timers else None

    @property
    def current(self) -> Optional[StagedTimer]:
        try:
            return timer_context.get()
        except:
            return None

    @property
    def timers(self) -> List[StagedTimer]:
        return self._timers

    @property
    def stages(self) -> List[str]:
        stages = set()

        for timer in self._timers:
            stages.update(timer.stages)

        return list(stages)

    @property
    def times(self) -> Dict[str, float]:
        all_times = self.all_times

        return {
            stage: sum(all_times[stage]) / len(all_times[stage])
            for stage in self.stages
        }

    @property
    def all_times(self) -> Dict[str, List[float]]:
        all_times = {stage: [] for stage in self.stages}

        for timer in self._timers:
            for stage, times in timer.all_times.items():
                all_times[stage].extend(times)

        return all_times

    @contextmanager
    def new_timer_context(self) -> StagedTimer:
        timer = StagedTimer(enabled=self.enabled)
        timer.start(InferenceStages.TOTAL_INFERENCE)

        if self.multi:
            self._timers.append(timer)
        else:
            self._timers = [timer]

        timer_context.set(timer)

        try:
            yield timer
        finally:
            timer.stop(InferenceStages.TOTAL_INFERENCE)
