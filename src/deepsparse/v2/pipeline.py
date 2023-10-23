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


from typing import Any, Dict, List, Optional, Union

from deepsparse.v2.operators import Operator
from deepsparse.v2.routers import Router
from deepsparse.v2.schedulers import OperatorScheduler, SchedulerGroup
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["Pipeline"]


class Pipeline(Operator):
    """
    Pipeline accepts a series of operators, schedulers, and a router. Calling a pipeline
    will use the router to run through all the defined operators.

    :param ops: Operators to run within the pipeline. Can either be a list of operators
    or dictionary of operators.
    :param router: A Router which dictates the next operator to call.
    :param schedulers: A list of schedulers to run operators.

    """

    def __init__(
        self,
        ops: Union[Dict[str, Operator], List[Operator]],
        router: Router,
        schedulers: List[OperatorScheduler],
        pipeline_state: PipelineState,
    ):

        self.ops = ops
        self.router = router
        self.schedulers = schedulers
        self.validate()
        self.pipeline_state = pipeline_state

        # SchedulerGroup handles running all schedulers in order of priority
        self._scheduler_group = SchedulerGroup(self.schedulers)

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
    ):
        """
        Run through the operators using the provided router and scheduler. Update the
        context to reflect each step of the router. The input to a given operator is the
        output of the previous operator.

        :param inp: input to the operator. expected to be of any type that is
        expected by the operator.
        :param context: context to store the current the inputs, outputs, and operator
        for each step of the router.

        """
        next_step = self.router.START_ROUTE
        while next_step != self.router.END_ROUTE:
            # Either a dictionary key or valid index
            operator = self.ops[next_step]
            print("BEFORE", next_step)

            output_future = self._scheduler_group.submit(
                operator=operator,
                operator_input=inp,
                context=context,
                pipeline_state=self.pipeline_state,
                inference_state=inference_state,
            )

            # wait for future to resolve
            operator_output, state_update = output_future.result()

            print("STATE UPDATE")
            print(state_update)
            # Update inference state
            inference_state.update_state(state_update)
            print(inference_state.current_state)

            # update context
            context.update(
                operator=operator,
                input=inp,
                output=operator_output,
            )

            next_step = self.router.next(next_step, self.ops, operator_output, inference_state)
            inp = operator_output

            print("NEXT", next_step)
        return operator_output, context

    def __call__(self, *args, return_context: bool = False, **kwargs):
        """
        :param return_context: if True, returns tuple of the pipeline output
            and entire context. Default False
        :return: output of the pipeline stages ran with the router for the given input
        """
        if len(args) > 1:
            raise ValueError(
                "Only 1 unnamed arg may be supplied to a Pipeline, the input expected"
                "for the first Operator."
            )
        if args and kwargs:
            raise ValueError(
                "Pipeline can only run either a in-line arguments or a "
                f"series of kwargs, found {len(args)} args and {len(kwargs)} kwargs"
            )

        pipeline_input = kwargs or args[0]

        # A new inference state is created for every pipeline inference run
        inference_state = InferenceState()
        inference_state.create_state({})

        pipeline_output, context = self.run(
            inp=pipeline_input,
            context=Context(),
            inference_state=inference_state,
        )

        if return_context:
            return pipeline_output, context

        return pipeline_output

    def validate(self):
        """
        Validate that compatability of the router and operators provided.
        """
        router_validation = self.router.validate(self.ops)

        if router_validation is False:
            # default error message
            stage_types = [type(stage) for stage in self.ops]
            raise ValueError(
                f"Invalid Router: {type(self.router)} for stages: {stage_types}"
            )
        elif isinstance(router_validation, str):
            raise ValueError(f"Invalid Router for stages: {router_validation}")
