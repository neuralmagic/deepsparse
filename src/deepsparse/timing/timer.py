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


__all__ = ["Timer"]


class Timer:
    """
    This object aggregates the durations
    (time deltas in seconds) of various components of the inference
    pipeline.
    Example flow:

    ```
    timer = Timer()

    timer.start("total_inference")

    timer.start("pre_process")
    do_something()
    timer.stop("pre_process")

    # time delta can we fetched directly after
    # record the stop time of the event
    pre_process_time_delta = timer.time_delta("pre_process")

    timer.start("engine_forward")
    do_something()
    timer.stop("engine_forward")

    timer.start("post_process")
    do_something()
    timer.stop("post_process")

    timer.stop("total_inference")

    # alternatively, time delta can we fetched later if convenient

    engine_forward_time_delta = timer.time_delta("engine_forward")
    post_process_time_delta = timer.time_delta("post_process")
    total_inference_time_delta = timer.time_delta("total_inference")
    ```
    The object may time the duration of an arbitrary number
    of events (phases). Choice of naming for phases is left
    for the user to decide.
    """

    def __init__(self):
        self._start_times = {}
        self._stop_times = {}

    def start(self, phase_name: str):
        """
        Collect the starting time of the phase

        :param phase_name: The name of an event (phase), which duration
            we are measuring
        """
        if phase_name in self._start_times:
            raise ValueError(
                f"Attempting to overwrite the start time of the phase: {phase_name}"
            )
        self._start_times[phase_name] = time.perf_counter()

    def stop(self, phase_name: str):
        """
        Collect the finish time of the phase

        :param phase_name: The name of an event (phase), which duration
            we are measuring
        """
        if phase_name not in self._start_times:
            raise ValueError(
                f"Attempting to grab the stop time of the phase: {phase_name},"
                f"but its start time is missing"
            )
        if phase_name in self._stop_times:
            raise ValueError(
                f"Attempting to overwrite the stop time of the phase: {phase_name}"
            )
        self._stop_times[phase_name] = time.perf_counter()

    def time_delta(self, phase_name: str) -> float:
        """
        If available, get the time delta (in seconds) of the event (phase).

        :param phase_name: The name of an event (phase), which time delta
            we want to get
        :return: the time delta (in seconds) of the specified phase
        """
        phase_start = self._start_times.get(phase_name)
        phase_stop = self._stop_times.get(phase_name)

        if not phase_start:
            raise ValueError(
                f"Attempting to fetch the duration of the phase: {phase_name}, but"
                f"its start time and stop time were not recorded"
            )
        elif not phase_stop:
            raise ValueError(
                f"Attempting to fetch the duration of the phase: {phase_name}, but"
                f"its stop time was not recorded"
            )
        else:
            return phase_stop - phase_start
