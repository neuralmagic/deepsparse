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
from typing import Any, List, Optional

from deepsparse.utils import SubGraph


__all__ = ["SubGraphExecutor"]


class SubGraphExecutor:
    def __init__(self, router, ops, run_next):
        self.router = router
        self.ops = ops
        self._run_next = run_next

    def start_subgraphs(
        self,
        sub_graph_inputs: List[Any],
        sub_graphs: List[SubGraph],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> List[Any]:
        """
        :param sub_graph_inputs: A list of inputs that should be passed to each
        subgraph. Each subgraph is given an element of the list as input to its
        first node.
        """
        for i in range(len(sub_graphs)):
            sub_graphs[i].output = self._run_next(
                sub_graph_inputs[i], sub_graphs[i].inf, sub_graphs[i].step, loop=loop
            )
        return sub_graphs

    def run_sub_graphs(self, sub_graphs: List[SubGraph]) -> List[Any]:
        """
        Run a list of sub_graphs asynchronously. Polls to identify the sub graph that is
        still running but has completed its current step. Schedules the next step
        subgraph step. This is repeated until all subgraphs have finished running and
        have reached their end step (stored in the Subgraph.end attribute).

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
                    self._run_next_step(sub_graph, operator_output)
        return [x.output for x in sub_graphs]

    async def run_sub_graphs_async(
        self,
        sub_graphs: List[SubGraph],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> List[Any]:

        # Execute all sub graphs until all graphs have been completed.
        while any(not x.completed for x in sub_graphs):
            for sub_graph in sub_graphs:
                if not sub_graph.completed:
                    # get the result for the completed operator; resolve its output
                    operator_output = await self._update_subgraph_async(sub_graph)
                    self._run_next_step(sub_graph, operator_output, loop=loop)

        return [x.output for x in sub_graphs]

    def _run_next_step(self, sub_graph, operator_output, loop=None):
        # determine the next step for the particular operator, using
        # its previous output and previously stored step
        next_step = self.router.next(sub_graph.step, self.ops, operator_output)
        # update the step
        sub_graph.step = next_step

        # store the output for the next step. If the next step is
        # end step, this particular route has completed. Simply
        # update the output value
        if next_step in sub_graph.end:
            sub_graph.output = operator_output
            sub_graph.completed = True
        else:
            sub_graph.output = self._run_next(
                inp=operator_output,
                inference_state=sub_graph.inf,
                next_step=next_step,
                loop=loop,
            )

    async def _update_subgraph_async(self, sub_graph):
        if isinstance(sub_graph.output, asyncio.Future):
            await sub_graph.output
        operator_output = sub_graph.output.result()
        operator_output = sub_graph.parse_output(operator_output)
        return operator_output

    def _update_subgraph(self, sub_graph):
        operator_output = sub_graph.output.result()
        operator_output = sub_graph.parse_output(operator_output)
        return operator_output
