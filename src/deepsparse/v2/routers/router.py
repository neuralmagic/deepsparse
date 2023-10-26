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


from typing import List, Tuple, Union

from deepsparse.v2.operators import Operator
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.utils import Context, OperatorSchema


__all__ = ["Router"]


class Router:
    """
    Routers must implement a run method which runs a series of operators
    for a pipeline for a given input. Base Router runs operators linearly
    in a series
    """

    @staticmethod
    def run(
        inp: OperatorSchema,
        operators: List[Operator],
        scheduler: OperatorScheduler,
    ) -> Tuple[OperatorSchema, Context]:
        """
        :param inp: input to the first operator of the series
        :param operators: list of operators to run
        :param scheduler: scheudler to submit operators to
        :return: final output of the operators
        """
        context = Context()

        # run operators linearly
        operator_input = inp
        for operator in operators:
            output_future = scheduler.submit(
                operator=operator, operator_input=operator_input, context=context
            )

            # wait for future to resolve
            operator_output = output_future.result()

            # update context
            context.update(
                operator=operator,
                input=operator_input,
                output=operator_output,
            )

            # previous output becomes next input
            operator_input = operator_output

        return operator_output, context

    @staticmethod
    def validate(operators: List[Operator]) -> Union[bool, str]:
        """
        :param operators: operators that this Router could potentially run over
        :return: True if this Router can run this series of operators. Base Router
            runs any series of operators that is non empty and whose input and output
            schemas align. If not valid, either False or an error string will be
            returned
        """
        if len(operators) < 1:
            return "No operators found"

        for idx in range(len(operators) - 1):
            current_output_schema = operators[idx].output_schema
            next_input_schema = operators[idx + 1].input_schema

            if current_output_schema is None or next_input_schema is None:
                # if no input/output schema defined, assume operator can run
                # without schema
                continue

            if current_output_schema != next_input_schema:
                return (
                    f"Operator at idx {idx}: {type(operators[idx])} has invalid "
                    f"output schema {current_output_schema} for next operator "
                    f"{type(operators[idx + 1])} which requires {next_input_schema}"
                )
