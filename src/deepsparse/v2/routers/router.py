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


from abc import abstractmethod
from typing import Dict, List, Union

from deepsparse.v2.operators import Operator


__all__ = ["Router", "LinearRouter"]


class Router:
    """
    Routers dicate the next operator to run. Each Router must implement a next function,
    which dictates the index or key of the next operator to run.

    :param start_route: the start index or key of the router
    :param end_route: the end index or key of the router

    """

    def __init__(self, end_route: Union[str, int], start_route: Union[str, int]):
        self.START_ROUTE = start_route
        self.END_ROUTE = end_route

    @abstractmethod
    def next(
        self, past: Union[str, int], ops: Union[List[Operator], Dict[str, Operator]]
    ) -> Union[str, int]:
        """
        Determines the index or dictionary key for the next operator which should run.

        :param past: the previous index or key. This should uniquely determine the next
        operator to run
        :param ops: list or dictionary of operators
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

    def next(self, past: int, ops: List[Operator]) -> int:
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
            # TODO: log
            return False

        for idx in range(len(operators) - 1):
            current_output_schema = operators[idx].output_schema
            next_input_schema = operators[idx + 1].input_schema

            if current_output_schema is None or next_input_schema is None:
                # if no input/output schema defined, assume operator can run
                # without schema
                continue

            if current_output_schema != next_input_schema:
                # TODO: Log error message below
                """
                return (
                    f"Operator at idx {idx}: {type(operators[idx])} has invalid "
                    f"output schema {current_output_schema} for next operator "
                    f"{type(operators[idx + 1])} which requires {next_input_schema}"
                )
                """
                return False
        return True
