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

from pydantic import BaseModel

from deepsparse.v2.operators import Operator
from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.routers import LinearRouter
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.utils import InferenceState, PipelineState


class IntSchema(BaseModel):
    value: int


class AddOneOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 1}


class AddTwoOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 2}


class TimedPipeline(Pipeline):
    def run(
        self,
        *args,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        **kwargs,
    ):
        """
        Check that timed info get saved in inference_state
        """

        func_nme = self.run.__name__

        self.timer_middleware.start_event(func_nme, inference_state)
        assert hasattr(inference_state, "timer")
        assert func_nme in getattr(inference_state, "timer").start_times

        time.sleep(1)

        self.timer_middleware.end_event(func_nme, inference_state)
        assert func_nme not in getattr(inference_state, "timer").start_times
        assert func_nme in getattr(inference_state, "timer").measurements

        # Only populate overall time to the middleware state
        assert func_nme not in self.timer_middleware.timer.start_times
        assert func_nme in self.timer_middleware.timer.measurements

        kwargs["inference_state"] = inference_state
        kwargs["pipeline_state"] = pipeline_state

        return super().run(
            *args,
            **kwargs,
        )

    def start_infernece_state_timer(self, state):
        self.timer_middleware.start_event("foo", state)

    def end_infernece_state_timer(self, state):
        self.timer_middleware.end_event("foo", state)


def test_timed_pipeline():
    AddThreePipeline = TimedPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    timer = AddThreePipeline.timer_middleware.timer
    assert "run" in timer.measurements

    time_delta = 0.01
    assert abs(1 - timer.measurements["run"]) < time_delta


def test_use_timer_from_inference_state():
    """
    inference state timings should be populated to the
    mmiddleware state using middleware end_event
    """
    AddThreePipeline = TimedPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    AddThreePipeline(pipeline_input)

    inference_state = InferenceState()
    inference_state.create_state({})

    name = "bar"
    AddThreePipeline.start_infernece_state_timer(inference_state)

    inference_state.timer.start(name)
    assert name in getattr(inference_state, "timer").start_times
    assert name not in getattr(inference_state, "timer").measurements

    time.sleep(1)
    inference_state.timer.end(name)

    assert name in getattr(inference_state, "timer").measurements
    assert name not in getattr(inference_state, "timer").start_times
    assert name not in getattr(AddThreePipeline.timer_middleware, "timer").measurements

    AddThreePipeline.end_infernece_state_timer(inference_state)

    assert name in getattr(inference_state, "timer").measurements
    assert name not in getattr(inference_state, "timer").start_times
    assert name in getattr(AddThreePipeline.timer_middleware, "timer").measurements
