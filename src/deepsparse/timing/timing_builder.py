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

from deepsparse.timing.timing_schema import InferenceTimingSchema


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
    builder.start()

    builder.{process_name}_start()
    something_happens()
    builder.{process_name}_complete()
    ...
    summary = builder.build()
    ```

    The object can be extended to aggregate further deltas.
    To extend, four things are required:
        - adding {process_name}_delta property (including the setter)
        - adding {process_name}_start method
        - adding {process_name}_complete method
        - adding the {process_name_delta} to the arguments collected
          inside the build() method.
    """

    def __init__(self):
        self.started = False

        # general placeholder for start time
        self._t0 = None
        # generate placeholder for completion time
        self._t1 = None

        self._pre_process_delta = None
        self._engine_forward_delta = None
        self._post_process_delta = None

    @property
    def t0(self) -> float:
        return self._t0

    @property
    def t1(self) -> float:
        return self._t1

    @property
    def pre_process_delta(self) -> float:
        return self._pre_process_delta

    @property
    def engine_forward_delta(self) -> float:
        return self._engine_forward_delta

    @property
    def post_process_delta(self) -> float:
        return self._post_process_delta

    @t0.setter
    def t0(self, value: float):
        if not self.started:
            raise ValueError(
                "Attempting to collect time information, "
                "but the TimingBuilder instance not started. "
                "Call start() method first"
            )
        if self.t1 is not None:
            raise ValueError(
                "Attempting to collect start time information, "
                "but the placeholder for completion time has not "
                "been reset. Make sure that the timing of the previous"
                "process has been completed"
            )
        self._t0 = value

    @t1.setter
    def t1(self, value: float):
        if not self.started:
            raise ValueError(
                "Attempting to collect time information, "
                "but the TimingBuilder instance not started. "
                "Call start() method first"
            )
        if self.t0 is None:
            raise ValueError(
                "Attempting to collect completion time information, "
                "but the placeholder for start time has not "
                "been reset. Make sure that the timing of the current"
                "process has been started"
            )
        self._t1 = value

    @pre_process_delta.setter
    def pre_process_delta(self, value: float):
        if self._pre_process_delta is not None:
            raise ValueError(
                "Attempting to overwrite the active time delta readout. "
                "This will be only possible once the build() method is called."
            )
        self._pre_process_delta = value

    @engine_forward_delta.setter
    def engine_forward_delta(self, value: float):
        if self._engine_forward_delta is not None:
            raise ValueError(
                "Attempting to overwrite the active time delta readout. "
                "This will be only possible once the build() method is called."
            )
        self._engine_forward_delta = value

    @post_process_delta.setter
    def post_process_delta(self, value: float):
        if self._post_process_delta is not None:
            raise ValueError(
                "Attempting to overwrite the active time delta readout. "
                "This will be only possible once the build() method is called."
            )
        self._post_process_delta = value

    def start(self):
        if self.started:
            raise ValueError("The TimingBuilder instance has been already started")
        self.started = True

    def build(self) -> InferenceTimingSchema:
        inference_timing_summary = InferenceTimingSchema(
            pre_process_delta=self.pre_process_delta,
            engine_forward_delta=self.engine_forward_delta,
            post_process_delta=self.post_process_delta,
            total_inference_delta=self.pre_process_delta
            + self.engine_forward_delta
            + self.post_process_delta,
        )
        self._cleanup()

        return inference_timing_summary

    def pre_process_start(self):
        self.t0 = time.time()

    def pre_process_complete(self):
        self.t1 = time.time()
        self.pre_process_delta = self.t1 - self.t0
        self._reset_time_counters()

    def engine_forward_start(self):
        self.t0 = time.time()

    def engine_forward_complete(self):
        self.t1 = time.time()
        self.engine_forward_delta = self.t1 - self.t0
        self._reset_time_counters()

    def post_process_start(self):
        self.t0 = time.time()

    def post_process_complete(self):
        self.t1 = time.time()
        self.post_process_delta = self.t1 - self.t0
        self._reset_time_counters()

    def _reset_time_counters(self):
        self._t0, self._t1 = None, None

    def _cleanup(self):
        self._t0, self._t1 = None, None

        self._pre_process_delta = None
        self._engine_forward_delta = None
        self._post_process_delta = None
