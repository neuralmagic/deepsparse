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

import asyncio
import copy
from asyncio import Future as AsyncioFuture
from concurrent.futures import Future
from functools import partial
from typing import Any, Callable, Dict, List, Union

from deepsparse.v2.operators import Operator
from deepsparse.v2.routers import Router
from deepsparse.v2.schedulers import OperatorScheduler, SchedulerGroup
from deepsparse.v2.utils import InferenceState, PipelineState


__all__ = ["Pipeline"]


class Pipeline(Operator):
    """
    Pipeline accepts a series of operators, schedulers, and a router. Calling a pipeline
    will use the router to run through all the defined operators. The operators should
    be implemented using the Operator class and each implemented operator should be
    responsible for a functional component of the pipelines. The flow of inputs/outputs
    between the operators and the steps in the pipeline should be defined by the router,
    (based off of the Router class), which dicates the next operator in the pipeline.
    Execution of the operators will be handled by the provided schedulers.

    :param ops: Operators to run within the pipeline. Can either be a list of operators
        or dictionary of operators.
    :param router: A Router which dictates the next operator to call.
    :param schedulers: A list of schedulers to run operators.
    :param pipeline_state: pipeline_state created during pipeline initialization

    """

    def __init__(
        self,
        ops: Union[Dict[str, Operator], List[Operator]],
        router: Router,
        schedulers: List[OperatorScheduler],
        pipeline_state: PipelineState = None,
    ):

        self.ops = ops
        self.router = router
        self.schedulers = schedulers
        self.pipeline_state = pipeline_state
        self.validate()

        self._scheduler_group = SchedulerGroup(self.schedulers)

    async def run_async(
        self,
        *args,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        index: int,
        **kwargs,
    ):
        """
        Run through the operators using the provided router and scheduler.
        The input to a given operator is the output of the previous operator.

        :param inference_state: inference_state for the pipeline.
        :param pipeline_state: pipeline_state for the pipeline. The values in the state
            are created during pipeline creation and are read-only during inference.
        """
        loop = asyncio.get_event_loop()

        next_step = self.router.START_ROUTE
        operator_output = None

        while next_step != self.router.END_ROUTE:
            # Either a dictionary key or valid index

            if next_step == self.router.SPLIT_ROUTE:
                operator_output = self._apply_split(operator_output, inference_state)
                next_step = self.router.route[self.router.JOIN_ROUTE]

            if next_step == self.router.START_ROUTE:
                outputs = self._run_next_step(
                    *args,
                    next_step=next_step,
                    func=self._scheduler_group.submit,
                    inference_state=inference_state,
                    operator=self.ops[next_step],
                    pipeline_state=pipeline_state,
                    loop=loop,
                    **kwargs,
                )
            else:
                outputs = self._run_next_step(
                    func=self._scheduler_group.submit,
                    input=operator_output,
                    next_step=next_step,
                    inference_state=inference_state,
                    operator=self.ops[next_step],
                    pipeline_state=pipeline_state,
                    loop=loop,
                )

            await outputs
            operator_output = outputs.result()

            if isinstance(operator_output, tuple):
                state_update = operator_output[-1]
                operator_output = operator_output[0]

            next_step = self.router.next(next_step, self.ops, operator_output)
            if state_update:
                inference_state.update_state(state_update)
        return operator_output

    def run(
        self,
        *args,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        **kwargs,
    ):
        """
        Run through the operators using the provided router and scheduler.
        The input to a given operator is the output of the previous operator.

        :param inference_state: inference_state for the pipeline.
        :param pipeline_state: pipeline_state for the pipeline. The values in the state
            are created during pipeline creation and are read-only during inference.
        """
        next_step = self.router.START_ROUTE
        operator_output = None

        while next_step != self.router.END_ROUTE:
            # NOTE: split_route should only appear after the start route node
            if next_step == self.router.SPLIT_ROUTE:
                operator_output = self._apply_split(operator_output, inference_state)
                next_step = self.router.route[self.router.JOIN_ROUTE]

            if next_step == self.router.START_ROUTE:
                outputs = self._run_next_step(
                    *args,
                    next_step=next_step,
                    func=self._scheduler_group.submit,
                    inference_state=inference_state,
                    operator=self.ops[next_step],
                    pipeline_state=pipeline_state,
                    **kwargs,
                )
            else:
                outputs = self._run_next_step(
                    func=self._scheduler_group.submit,
                    input=operator_output,
                    next_step=next_step,
                    inference_state=inference_state,
                    operator=self.ops[next_step],
                    pipeline_state=pipeline_state,
                )
            next_step, operator_output, state_update = outputs
            if state_update:
                inference_state.update_state(state_update)
        return operator_output

    def __call__(self, *args, **kwargs):
        """
        Consolidate any provided inference_state or pipeline_state objects and pass
        any other operator inputs to run().

        :return: output of the pipeline operators ran with the router for the given
            input
        """
        if kwargs.get("inference_state"):
            inference_state = kwargs.pop("inference_state")
        else:
            inference_state = InferenceState()
            inference_state.create_state({})

        if "pipeline_state" in kwargs:
            self.pipeline_state = kwargs.get("pipeline_state")

        kwargs["inference_state"] = inference_state
        kwargs["pipeline_state"] = self.pipeline_state

        return self.run(*args, **kwargs)

    def expand_inputs(self, *args, **kwargs):
        """
        Generic function to handle expanding values.
        """
        raise NotImplementedError(
            "This function should be implemented for any router with split or join"
            "nodes. expand_inputs will be called prior to the split node (stored in "
            "the router's SPLIT_ROUTE attribute), expanding outputs for each output "
            "such that there is a batch size of one per thread."
        )

    def condense_inputs(self, *args, **kwargs):
        """
        Generic function to handle condensing values.
        """
        raise NotImplementedError(
            "This function should be implemented for any router with split or join "
            "nodes. condense_inputs will be called after the join node (stored in the "
            "router's JOIN_ROUTE attribute), condensing outputs from multiple threads."
        )

    def validate(self):
        """
        Validate that compatability of the router and operators provided.
        """
        router_validation = self.router.validate(self.ops)

        if router_validation is False:
            # default error message
            op_types = [type(op) for op in self.ops]
            raise ValueError(f"Invalid Router: {type(self.router)} for ops: {op_types}")
        elif isinstance(router_validation, str):
            raise ValueError(f"Invalid Router for operators: {router_validation}")

    def _apply_split(self, inp: Any, inference_state: InferenceState):
        """
        Split inputs using the pipeline's expand_inputs function. Inputs are split
        into a batch size of one when a SPLIT_ROUTE node is found in a given pipeline's
        provided router. The split batches are run asynchronously and then joined when
        a JOIN_ROUTE node is found, using the pipeline's condense_inputs function.
        """

        batches, orig_batch_size = self.expand_inputs(inp, 1)
        run_with_state = partial(
            self._run_sequential,
            pipeline_state=self.pipeline_state,
            start=self.router.route[self.router.SPLIT_ROUTE],
            end=self.router.JOIN_ROUTE,
        )
        inference_state_list = [
            copy.deepcopy(inference_state) for x in range(len(batches))
        ]
        futures = self._scheduler_group.map(
            batches,
            inference_state_list,
            func=run_with_state,
        )
        return self.condense_inputs([x.result() for x in futures])

    def _run_next_step(
        self,
        *args,
        func: Callable,
        next_step: Union[str, int],
        input: Any = None,
        **kwargs,
    ):
        """
        Generic function to run a given func, process the output and determine the next
        step.
        """
        if input:
            operator_output = (
                func(*args, **kwargs, **input)
                if isinstance(input, dict)
                else func(input, *args, **kwargs)
            )
        else:
            operator_output = func(*args, **kwargs)

        if isinstance(operator_output, AsyncioFuture):
            return operator_output

        if isinstance(operator_output, Future):
            operator_output = operator_output.result()

        state_update = None
        if isinstance(operator_output, tuple):
            state_update = operator_output[-1]
            operator_output = operator_output[0]

        next_step = self.router.next(next_step, self.ops, operator_output)
        return next_step, operator_output, state_update

    def _run_sequential(
        self,
        inp: Any,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        start: str,
        end: str,
    ):
        next_step = start
        while next_step != end:
            outputs = self._run_next_step(
                func=self.ops[next_step],
                next_step=next_step,
                input=inp,
                pipeline_state=pipeline_state,
                inference_state=inference_state,
            )
            next_step, operator_output, state_update = outputs
            if state_update:
                inference_state.update_state(state_update)
            inp = operator_output
        return inp
