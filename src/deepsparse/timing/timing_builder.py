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
from typing import Dict


__all__ = ["TimingBuilder"]


class TimingBuilder:
    """
    This object aggregates the durations
    (time deltas in seconds) of various components of the inference
    pipeline.
    Once all the desired time deltas are aggregated, the
    object may build the data structure that summarizes the
    collected information.

    Example flow:

    ```
    builder = TimingBuilder()

    builder.start("total_inference")

    builder.start("pre_process")
    do_something()
    builder.stop("pre_process")

    builder.start("engine_forward")
    do_something()
    builder.stop("engine_forward")

    builder.start("post_process")
    do_something()
    builder.stop("post_process")

    builder.stop("total_inference")
    ...
    summary = builder.build()
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

    def build(self) -> Dict[str, float]:
        """
        Aggregate the collected measurements and return them as a
        dictionary
        :return: Mapping from the phase name to the phase duration in seconds.
        """
        return self._compute_time_deltas()

    def _compute_time_deltas(self) -> Dict[str, float]:
        deltas = {}
        phases = self._start_times.keys()
        for phase_name in phases:
            phase_start = self._start_times[phase_name]
            phase_stop = self._stop_times[phase_name]
            time_delta = phase_stop - phase_start
            deltas[phase_name] = time_delta
        return deltas
