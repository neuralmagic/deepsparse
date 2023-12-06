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


import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from deepsparse.v2.operators import Operator


_LOGGER = logging.getLogger(__name__)

__all__ = ["Router", "LinearRouter", "GraphRouter"]


class Router:
    """
    Routers dicate the next operator to run. Each Router must implement a next function,
    which dictates the index or key of the next operator to run.

    :param start_route: the start index or key of the router
    :param end_route: the end index or key of the router
    :param route: the route that the router has to traverse through

    """

    def __init__(
        self,
        end_route: Union[str, int],
        start_route: Union[str, int],
        route: Optional[Dict] = None,
        split_route: str = "SPLIT",
        join_route: str = "JOIN",
    ):
        self.START_ROUTE = start_route
        self.END_ROUTE = end_route
        self.SPLIT_ROUTE = split_route
        self.JOIN_ROUTE = join_route
        self.route = route

    @abstractmethod
    def next(
        self,
        past: Union[str, int],
        ops: Optional[Union[List[Operator], Dict[str, Operator]]],
        inp: Optional[Any],
    ) -> Union[str, int]:
        """
        Determines the index or dictionary key for the next operator which should run.

        :param past: the previous index or key. This should uniquely determine the next
            operator to run
        :param ops: list or dictionary of operators
        :param inp: operator input
        :returns: the next index or dictionary key for the next operator to run
        """
        raise NotImplementedError

    def yaml(self):
        pass

    def json(self):
        pass


class LinearRouter(Router):
    """
    LinearRouterruns a list of Operators in sequential order. end_route should
    be the length of the list and the start_route should be the start index.
    """

    def __init__(self, end_route: int, start_route: int = 0):
        super().__init__(end_route=end_route, start_route=start_route)
        _LOGGER.warn("SPLIT and JOIN are not yet supported for the LinearRouter.")

    def next(
        self, past: int, ops: Optional[List[Operator]] = None, inp: Optional[Any] = None
    ) -> int:
        new_index = past + 1
        if new_index < self.END_ROUTE:
            return new_index
        return self.END_ROUTE

    @staticmethod
    def validate(operators: List[Operator]) -> bool:
        """
        :param operators: operators that this Router could potentially run over
        :return: True if this Router can run this series of operators. Base Router
            runs any series of operators that is non empty and whose input and output
            schemas align. If not valid, either False or an error string will be
            returned
        """
        if len(operators) < 1:
            _LOGGER.info("No operators provided")
            return False

        for idx in range(len(operators) - 1):
            current_output_schema = operators[idx].output_schema
            next_input_schema = operators[idx + 1].input_schema

            if current_output_schema is None or next_input_schema is None:
                # if no input/output schema defined, assume operator can run
                # without schema
                continue

            if current_output_schema != next_input_schema:
                _LOGGER.info(
                    f"Operator at idx {idx}: {type(operators[idx])} has invalid "
                    f"output schema {current_output_schema} for next operator "
                    f"{type(operators[idx + 1])} which requires {next_input_schema}"
                )
                return False
        return True


class GraphRouter(Router):
    """
    Router for a DAG. Expects graphs be presented in the form of a dictionary, where
    keys are the nodes of the graph and the values are the connected nodes. For
    nodes with multiple ouput edges, all the nodes will be visited and the first node
    where `can_operate` returns True will run. Paths should be deterministic.
    """

    def __init__(self, end_route: str, start_route: str, route: Dict, **kwargs):
        super().__init__(
            end_route=end_route, start_route=start_route, route=route, **kwargs
        )

    def next(
        self,
        past: str,
        ops: Dict[str, Operator],
        inp: Any,
    ) -> int:
        node = past
        if isinstance(self.route[node], str):
            return self.route[node]
        else:
            for neighbour_node in self.route[node]:
                neighbour_node_op = ops[neighbour_node]
                if neighbour_node_op.can_operate(inp):
                    return neighbour_node
            raise ValueError("Cannot operate on any of the nodes")

    @staticmethod
    def validate(ops) -> bool:
        # TODO: still needs to be implemented for the GraphRouter
        pass
