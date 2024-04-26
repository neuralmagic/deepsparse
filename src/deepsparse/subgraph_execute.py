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
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional

from pydantic import BaseModel, Field

from deepsparse.operators import Operator
from deepsparse.routers import Router
from deepsparse.utils import SubGraph


__all__ = ["SubGraphExecutor", "StreamingOutput"]


class StreamingOutput(BaseModel):
    """
    Helper object to store the output of a streaming operator. Facilitates
    returning data to be used in the next step of the pipeline and yielding
    the data immediately from the pipeline.
    """

    data_to_return: Any = Field(
        None,
        description="Data that should be returned to be used in the next pipeline step",
    )
    data_to_yield: Any = Field(
        None, description="Data that should be yielded to the user"
    )


class SubGraphExecutor:
    """
    An executor class created to run the steps of a Subgraph. The subgraph can run
    synchronously, asynchronously and also produce a generator for each case.
    """

    def start_subgraphs(
        self,
        func: Callable,
        sub_graph_inputs: List[Any],
        sub_graphs: List[SubGraph],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> List[Any]:
        """
        Start the SubGraph execution. This step should run before any of the run_
        executions functions.

        :param func: callable to schedule next step
        :param sub_graph_inputs: A list of inputs that should be passed to each
        subgraph. Each subgraph is given an element of the list as input to its
        first node.
        :param sub_graphs: A list of Subgraph objects. Each stores the relevant
            execution information for the particular subgraph, such as its current step
            in the sub graph, inference state, output, and end step.
        :param loop: Optional asyncio.AbstractEventLoop needed for async scheduling

        :returns: list of subgraphs. initial operator step in each subgraph should be
            scheduled.
        """
        for i in range(len(sub_graphs)):
            sub_graphs[i].output = func(
                sub_graph_inputs[i], sub_graphs[i].inf, sub_graphs[i].step, loop=loop
            )
        return sub_graphs

    def run_sub_graphs(
        self,
        router: Router,
        ops: Dict[str, Operator],
        func: Callable,
        sub_graphs: List[SubGraph],
    ) -> List[Any]:
        """
        Run a list of sub_graphs. Polls to identify the sub graph that is
        still running but has completed its current step. Schedules the next step
        subgraph step. This is repeated until all subgraphs have finished running and
        have reached their end step (stored in the Subgraph.end attribute).

        :params router: router to indentify the next step
        :param ops: a list or dictionary of operators to determine the next step
        :params func: callable to schedule next step
        :param sub_graphs: A list of Subgraph objects. Each stores the relevant
            execution information for the particular subgraph, such as its current step
            in the sub graph, inference state, output, and end step.

        :returns: a list of outputs for all the completed Subgraph objects. Returned
        in the same order that the subgraphs were passed to the function.
        """

        # Execute all sub graphs until all graphs have been completed.
        while any(not x.completed for x in sub_graphs):
            for sub_graph in sub_graphs:
                if not sub_graph.completed:
                    # get the result for the completed operator; resolve its output
                    operator_output = self._update_subgraph(sub_graph)
                    self._run_next_step(router, ops, func, sub_graph, operator_output)

        return [x.output for x in sub_graphs]

    async def run_sub_graphs_async(
        self,
        router: Router,
        ops: Dict[str, Operator],
        func: Callable,
        sub_graphs: List[SubGraph],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> List[Any]:

        """
        :param router: router to indentify the next step
        :param ops: a list or dictionary of operators to determine the next step
        :param func: callable to schedule next step
        :param sub_graphs: A list of Subgraph objects. Each stores the relevant
            execution information for the particular subgraph, such as its current step
            in the sub graph, inference state, output, and end step.

        :returns: a list of outputs for all the completed Subgraph objects. Returned
        in the same order that the subgraphs were passed to the function.
        """

        while any(not x.completed for x in sub_graphs):
            for sub_graph in sub_graphs:
                if not sub_graph.completed:
                    operator_output = await self._update_subgraph_async(sub_graph)
                    self._run_next_step(
                        router, ops, func, sub_graph, operator_output, loop=loop
                    )

        return [x.output for x in sub_graphs]

    def run_sub_graphs_generator(
        self,
        router: Router,
        ops: Dict[str, Operator],
        func: Callable,
        sub_graphs: List[SubGraph],
    ) -> Generator:
        """
        Applies the same logic as run_sub_graphs but instead returns a Generator.
        """
        while any(not x.completed for x in sub_graphs):
            for sub_graph in sub_graphs:
                output_to_yield = None
                if not sub_graph.completed:
                    operator_output = self._update_subgraph(sub_graph)
                    operator_output, output_to_yield = self._parse_streaming_output(
                        operator_output
                    )
                    self._run_next_step(router, ops, func, sub_graph, operator_output)
                    if output_to_yield:
                        yield output_to_yield, sub_graph.step, operator_output, sub_graph.inf  # noqa: E501

    async def run_sub_graphs_async_generator(
        self,
        router: Router,
        ops: Dict[str, Operator],
        func: Callable,
        sub_graphs: List[SubGraph],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> AsyncGenerator:

        """
        Applies the same logic as run_sub_graphs_async but instead returns an
        AsyncGenerator.
        """
        while any(not x.completed for x in sub_graphs):
            for sub_graph in sub_graphs:
                output_to_yield = None
                if not sub_graph.completed:
                    operator_output = await self._update_subgraph_async(sub_graph)
                    operator_output, output_to_yield = self._parse_streaming_output(
                        operator_output
                    )
                    self._run_next_step(
                        router, ops, func, sub_graph, operator_output, loop=loop
                    )
                    if output_to_yield:
                        yield output_to_yield, sub_graph.step, operator_output, sub_graph.inf  # noqa: E501

    def _run_next_step(
        self,
        router: Router,
        ops: Dict[str, Operator],
        func: Callable,
        sub_graph: SubGraph,
        operator_output: Any,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Determine the next operator step, using its previous output and previously
        stored step. Schedule the next step to run or set as completed if on end step.

        :param router: router to indentify the next step
        :param ops: a list or dictionary of operators to determine the next step
        :param func: callable to schedule next step
        :param subg_graph: subgraph to determine the next step for and schedule
        :param operator_output: output produced by the subgraph
        :param loop: Optional asyncio.AbstractEventLoop needed for async scheduling
        """

        next_step = router.next(sub_graph.step, ops, operator_output)
        sub_graph.step = next_step

        if next_step in sub_graph.end:
            sub_graph.output = operator_output
            sub_graph.completed = True
        else:
            sub_graph.output = func(
                inp=operator_output,
                inference_state=sub_graph.inf,
                next_step=next_step,
                loop=loop,
            )

    async def _update_subgraph_async(self, sub_graph: SubGraph):
        """
        Get the result for the current operator running in the sub_graph and parse its
        output. Await the output if running async.

        :params sub_graph: SubGraph object
        """
        await sub_graph.output
        operator_output = sub_graph.output.result()
        operator_output = sub_graph.parse_output(operator_output)
        return operator_output

    def _update_subgraph(self, sub_graph: SubGraph):
        """
        Get the result for the current operator running in the sub_graph and parse its
        output.

        :params sub_graph: SubGraph object
        """
        operator_output = sub_graph.output.result()
        operator_output = sub_graph.parse_output(operator_output)
        return operator_output

    def _parse_streaming_output(self, operator_output: Any):
        """
        Determine if the output produced is a StreamingOutput. If that is the case,
        parse the output to return the value that should be passed to the next operator
        and the value that should be yielded.

        :params operator_output: output produced by the operator within a subgraph
        :returns: parsed operator output and if given, the output to yield
        """
        output_to_yield = None
        if isinstance(operator_output, StreamingOutput):
            output_to_yield = operator_output.data_to_yield
            operator_output = operator_output.data_to_return
        return operator_output, output_to_yield
