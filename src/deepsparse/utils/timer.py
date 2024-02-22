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

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional


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
        """
        Provide a string representation of the StagedTimer object.

        :return: a string representing the timer object with its times.
        """
        return f"StagedTimer({self.times})"

    @property
    def stages(self) -> List[str]:
        """
        Get the stages for the timer object.

        :return: list of stages as strings.
        """
        return list(self._staged_start_times.keys())

    @property
    def times(self) -> Dict[str, float]:
        """
        Get the average time for each stage.

        :return: a dictionary with stage names as keys and their average time as values.
        """
        return {stage: self.stage_average_time(stage) for stage in self.stages}

    @property
    def all_times(self) -> Dict[str, List[float]]:
        """
        Get the list of times for each stage.

        :return: a dictionary with stages as keys and their list of times as values.
        """
        return {stage: self.stage_times(stage) for stage in self.stages}

    def clear(self):
        """
        Clear all the stored start and stop times for all stages.
        """
        self._staged_start_times.clear()
        self._staged_stop_times.clear()

    def has_stage(self, stage: str) -> bool:
        """
        Check if a stage exists in the timer.

        :param stage: the name of the stage to check.
        :return: True if the stage exists, False otherwise.
        """
        return stage in self.stages

    @contextmanager
    def time(self, stage: str):
        """
        Context Manager to record the time for a stage in the given context

        example:
        ```
        with timer.time(STAGE_NAME):
            # do something...
        ```

        :param stage: the name of the stage to time
        """
        self.start(stage)
        yield
        self.stop(stage)

    def start(self, stage: str):
        """
        Start the timer for a specific stage. If the stage doesn't exist,
        it's added to the timer.

        :param stage: the name of the stage to start.
        :raises ValueError: if trying to start a stage before a previous one
                            has been stopped.
        """
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
        """
        Stop the timer for a specific stage.

        :param stage: the name of the stage to stop.
        :raises ValueError: if trying to stop a stage that has not been started
                            or if trying to stop a stage before a previous one
                            has been started.
        """
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
        """
        Get the list of time deltas for a specific stage.

        :param stage: the name of the stage to get time deltas for.
        :return: a list of time deltas.
        :raises ValueError: if trying to get time deltas for a stage that has not been
                            started or a stage that has not been stopped.
        """
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
        """
        Get the average time for a specific stage.

        :param stage: the name of the stage to get the average time for.
        :return: the average time for the specified stage.
        """
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
        """
        Provide a string representation of the TimerManager object.

        :return: a string representing the timer manager object with its times.
        """
        return f"TimerManager({self.times})"

    @property
    def latest(self) -> Optional[StagedTimer]:
        """
        Get the latest created StagedTimer.

        :return: the latest created StagedTimer object or None if no timers are present.
        """
        return self._timers[-1] if self._timers else None

    @property
    def current(self) -> Optional[StagedTimer]:
        """
        Get the current active StagedTimer in the context.

        :return: the current active StagedTimer object in the context or
                 None if no timers are active.
        """
        try:
            return timer_context.get()
        except LookupError:
            # no timer in context, return None
            return None

    @property
    def timers(self) -> List[StagedTimer]:
        """
        Get the list of all StagedTimer objects.

        :return: a list of all StagedTimer objects.
        """
        return self._timers

    @property
    def stages(self) -> List[str]:
        """
        Get the unique list of stages from all StagedTimer objects.

        :return: a list of unique stages.
        """
        stages = set()

        for timer in self._timers:
            stages.update(timer.stages)

        return list(stages)

    @property
    def times(self) -> Dict[str, float]:
        """
        Get the average time for each stage across all StagedTimer objects.

        :return: a dictionary with stage names as keys and their average time as values.
        """
        all_times = self.all_times

        return {
            stage: sum(all_times[stage]) / len(all_times[stage])
            for stage in self.stages
        }

    @property
    def all_times(self) -> Dict[str, List[float]]:
        """
        Get the list of times for each stage across all StagedTimer objects.

        :return: a dictionary with stages as keys and their list of times as values.
        """
        all_times = {stage: [] for stage in self.stages}

        for timer in self._timers:
            for stage, times in timer.all_times.items():
                all_times[stage].extend(times)

        return all_times

    def current_or_new(self) -> StagedTimer:
        """
        Return the current timer if there is one, otherwise return a new one.
        """
        if self.current:
            return self.current
        else:
            with self.new_timer_context(total_inference=False) as timer:
                return timer

    def clear(self):
        for t in self._timers:
            t.clear()

    @contextmanager
    def new_timer_context(self, total_inference: bool = True) -> StagedTimer:
        """
        Create a new StagedTimer object and set it as the current context.

        :param total_inference: if True, measures the entire context as total inference
            automatically and assumes this is the main inference thread. if False,
            assumes this is not the main inference thread and will not overwrite
            any other timers in non-multi/benchmark mode. Default True
        :return: the new StagedTimer object.
        """
        timer = StagedTimer(enabled=self.enabled)

        if total_inference:
            timer.start(InferenceStages.TOTAL_INFERENCE)

        if self.multi or not total_inference:
            self._timers.append(timer)
        else:
            self._timers = [timer]

        timer_context.set(timer)
        yield timer
        if total_inference:
            timer.stop(InferenceStages.TOTAL_INFERENCE)
