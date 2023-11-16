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


import copy
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Union

from deepsparse.v2.operators import EngineOperator, Operator
from deepsparse.v2.routers import Router
from deepsparse.v2.schedulers import (
    ContinuousBatchingScheduler,
    OperatorScheduler,
    SchedulerGroup,
)
from deepsparse.v2.utils import InferenceState, PipelineState, SplitRoute


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
        continuous_batching_scheduler: ContinuousBatchingScheduler,
        pipeline_state: PipelineState = None,
    ):

        self.ops = ops
        self.router = router
        self.schedulers = schedulers
        self.pipeline_state = pipeline_state
        self._continuous_batching_scheduler = continuous_batching_scheduler
        self.validate()

        self._scheduler_group = SchedulerGroup(self.schedulers)

    def _run_next(
        self,
        inp: Any,
        inference_state: InferenceState,
        next_step: str,
    ):
        if (
            isinstance(self.ops[next_step], EngineOperator)
            and self._continuous_batching_scheduler
        ):
            func = self._continuous_batching_scheduler.submit
            inp = self.ops[next_step].input_schema(**inp)
        else:
            func = self._scheduler_group.submit

        return self._run_func(
            func=func,
            operator=self.ops[next_step],
            inp=inp,
            pipeline_state=self.pipeline_state,
            inference_state=inference_state,
        )

    def _apply_split(self, inp: Any, inference_state: InferenceState):
        """
        Split inputs using the pipeline's expand_inputs function. Inputs are split
        into a batch size of one when a SPLIT_ROUTE node is found in a given pipeline's
        provided router. The split batches are run asynchronously and then joined when
        a JOIN_ROUTE node is found, using the pipeline's condense_inputs function.
        """

        batches, orig_batch_size = self.expand_inputs(inp, 1)

        # Create a list of SplitRoutes, per batch size 1
        # Each SplitRoute object holds information about the particular path it
        # follows. All start at the same step defined by SPLIT_ROUTE and start
        # with the same inference_state.
        split_routes = [
            SplitRoute(
                inf=copy.deepcopy(inference_state),
                step=self.router.route[self.router.SPLIT_ROUTE],
            )
            for i in range(len(batches))
        ]

        # Start each SPLIT_ROUTE; store the Future for each.
        for i in range(len(split_routes)):
            split_routes[i].output = self._run_next(
                batches[i], split_routes[i].inf, split_routes[i].step
            )

        # Execute all split batches until all split batches have been completed.
        while True:
            for split_route in split_routes:
                if isinstance(split_route.output, Future) and split_route.output.done():
                    # get the result for the completed operator; resolve its output
                    operator_output = split_route.output.result()
                    operator_output = split_route.parse_output(operator_output)

                    # determine the next step for the particular operator, using
                    # its previous output and previously stored step
                    next_step = self.router.next(
                        split_route.step, self.ops, operator_output
                    )
                    # update the step
                    split_route.step = next_step

                    # store the output for the next step. If the next step is
                    # JOIN_ROUTE note, this particular route has completed. Simply
                    # update the output value
                    if next_step == self.router.JOIN_ROUTE:
                        split_route.output = operator_output
                    else:
                        split_route.output = self._run_next(
                            inp=operator_output,
                            inference_state=split_route.inf,
                            next_step=next_step,
                        )
                    break

            # keep running until all split routes have completed.
            if not any(isinstance(x.output, Future) for x in split_routes):
                break

        return self.condense_inputs([x.output for x in split_routes])

    def _run_func(
        self,
        *args,
        func: Callable,
        inp: Any = None,
        **kwargs,
    ):
        """
        Generic function to run a given Callable.
        """
        if inp:
            operator_output = (
                func(*args, **kwargs, **inp)
                if isinstance(inp, dict)
                else func(inp, *args, **kwargs)
            )
        else:
            operator_output = func(*args, **kwargs)
        return operator_output

    def run(
        self,
        *args,
        inference_state: InferenceState,
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
                if operator_output is None:
                    raise ValueError(
                        f"{self.router.SPLIT_ROUTE} should appear after "
                        f"{self.ROUTER.START_ROUTE}"
                    )

                operator_output = self._apply_split(operator_output, inference_state)
                next_step = self.router.route[self.router.JOIN_ROUTE]
                if next_step == self.router.END_ROUTE:
                    return operator_output

            if next_step == self.router.START_ROUTE:
                outputs = self._run_func(
                    *args,
                    func=self._scheduler_group.submit,
                    operator=self.ops[next_step],
                    inference_state=inference_state,
                    pipeline_state=self.pipeline_state,
                    **kwargs,
                )
            else:
                outputs = self._run_next(
                    inp=operator_output,
                    next_step=next_step,
                    inference_state=inference_state,
                )

            operator_output = outputs.result()

            if isinstance(operator_output, tuple):
                state_update = operator_output[-1]
                operator_output = operator_output[0]
                inference_state.update_state(state_update)

            next_step = self.router.next(next_step, self.ops, operator_output)
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

        kwargs["inference_state"] = inference_state

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
