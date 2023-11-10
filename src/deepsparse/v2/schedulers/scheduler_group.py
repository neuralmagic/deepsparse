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
        *args,
        operator: Operator,
        loop: Any = None,
        **kwargs,
    ) -> Future:
        """
        :param operator: operator to run
        :return: future referencing the asynchronously run output of the operator
        """
        for scheduler in self.schedulers:
            if scheduler.can_process(
                *args,
                operator=operator,
                **kwargs,
            ):
                if loop:
                    return scheduler.async_run(
                        *args, operator=operator, loop=loop, **kwargs
                    )
                return scheduler.submit(
                    *args,
                    operator=operator,
                    **kwargs,
                )
