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

from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.utils import InferenceState, PipelineState


class TimedMiddlewarePipeline(Pipeline):
    def run(
        self,
        *args,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        **kwargs,
    ):
        self._save_run_time_to_middleware_state("foo", 1)
        self._save_run_time_to_middleware_state("bar", 0.5)

        kwargs["inference_state"] = inference_state
        kwargs["pipeline_state"] = pipeline_state

        time_delta = 0.005

        assert (
            abs(1 - getattr(self.timer_middleware, "timer").measurements["foo"])
            < time_delta
        )
        assert (
            abs(0.5 - getattr(self.timer_middleware, "timer").measurements["bar"])
            < time_delta
        )

        return super().run(
            *args,
            **kwargs,
        )

    def _save_run_time_to_middleware_state(self, key: str, run_time: int):
        self.timer_middleware.start_event(key)
        assert hasattr(self.timer_middleware, "timer")
        assert key in getattr(self.timer_middleware, "timer").start_times

        time.sleep(run_time)

        self.timer_middleware.end_event(key)
        assert key not in getattr(self.timer_middleware, "timer").start_times
        assert key in getattr(self.timer_middleware, "timer").measurements


class TimedInferenceStatePipeline(Pipeline):
    def run(
        self,
        *args,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        **kwargs,
    ):
        self._save_run_time_to_inference_state("fp32", 1, inference_state)
        self._save_run_time_to_inference_state("int4", 0.5, inference_state)

        kwargs["inference_state"] = inference_state
        kwargs["pipeline_state"] = pipeline_state

        time_delta = 0.005
        assert (
            abs(1 - getattr(inference_state, "timer").measurements["fp32"]) < time_delta
        )
        assert (
            abs(0.5 - getattr(inference_state, "timer").measurements["int4"])
            < time_delta
        )

        return super().run(
            *args,
            **kwargs,
        )

    def _save_run_time_to_inference_state(
        self, key: str, run_time: int, state: InferenceState
    ):
        self.timer_middleware.start_event(key, state)
        assert hasattr(state, "timer")
        assert key in getattr(state, "timer").start_times

        time.sleep(run_time)

        self.timer_middleware.end_event(key, state)
        assert key not in getattr(state, "timer").start_times
        assert key in getattr(state, "timer").measurements
        assert key not in getattr(self.timer_middleware, "timer").measurements


class TimedInferenceStateMiddlewarePipeline(Pipeline):
    def run(
        self,
        *args,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        **kwargs,
    ):
        self._save_run_time_to_middleware_state("foo", 1)
        self._save_run_time_to_middleware_state("bar", 0.5)

        self._save_run_time_to_inference_state("fp32", 1, inference_state)
        self._save_run_time_to_inference_state("int4", 0.5, inference_state)

        kwargs["inference_state"] = inference_state
        kwargs["pipeline_state"] = pipeline_state

        rtn = super().run(
            *args,
            **kwargs,
        )

        self._save_inference_time_to_middleware_state("op1", inference_state)

        return rtn

    def _save_inference_time_to_middleware_state(self, key: str, state: InferenceState):
        assert key not in getattr(self.timer_middleware, "timer").measurements

        self.timer_middleware.update_middleware_timer(key, state)
        assert key in getattr(self.timer_middleware, "timer").measurements
        for updated_key, updated_value in state.timer.measurements.items():
            assert (
                updated_key in getattr(self.timer_middleware, "timer").measurements[key]
            )
            assert (
                updated_value
                == getattr(self.timer_middleware, "timer").measurements[key][
                    updated_key
                ]
            )

    def _save_run_time_to_middleware_state(self, key: str, run_time: int):
        self.timer_middleware.start_event(key)
        assert hasattr(self.timer_middleware, "timer")
        assert key in getattr(self.timer_middleware, "timer").start_times

        time.sleep(run_time)

        self.timer_middleware.end_event(key)
        assert key not in getattr(self.timer_middleware, "timer").start_times
        assert key in getattr(self.timer_middleware, "timer").measurements

    def _save_run_time_to_inference_state(
        self, key: str, run_time: int, state: InferenceState
    ):
        self.timer_middleware.start_event(key, state)
        assert hasattr(state, "timer")
        assert key in getattr(state, "timer").start_times

        time.sleep(run_time)

        self.timer_middleware.end_event(key, state)
        assert key not in getattr(state, "timer").start_times
        assert key in getattr(state, "timer").measurements
        assert key not in getattr(self.timer_middleware, "timer").measurements
