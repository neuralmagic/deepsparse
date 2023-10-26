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


from concurrent.futures import Future
from typing import Any, List

from deepsparse.v2.operators import Operator
from deepsparse.v2.schedulers.scheduler import OperatorScheduler
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["SchedulerGroup"]


class SchedulerGroup(OperatorScheduler):
    """
    Wrapper for a series of schedulers. Runs submitted operators on the first
    scheduler that can process a given input

    :param schedulers: list of schedulers to pass operators to
    """

    def __init__(self, schedulers: List[OperatorScheduler]):
        self.schedulers = schedulers

    def submit(
        self,
        operator: Operator,
        operator_input: Any,
        context: Context,
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ) -> Future:
        """
        :param operator: operator to run
        :param operator_input: input schema to the operator
        :param context: context of already run operators
        :return: future referencing the asynchronously run output of the operator
        """
        for scheduler in self.schedulers:
            if scheduler.can_process(operator, operator_input):
                return scheduler.submit(
                    operator, operator_input, context, pipeline_state, inference_state
                )

    def can_process(self, operator: Operator, operator_input: Any) -> bool:
        """
        :param operator: operator to check
        :param operator_input: operator_input to check
        :return: True if this Operator can process the given operator and input.
            SchedulerGroup always returns True
        """
        return any(
            scheduler.can_process(operator, operator_input)
            for scheduler in self.schedulers
        )