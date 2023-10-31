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


from concurrent.futures import Future, ThreadPoolExecutor

from deepsparse.v2.operators import Operator


__all__ = ["OperatorScheduler"]


class OperatorScheduler:
    """
    OperatorSchedulers should implement a `submit` function that asynchronously
    runs an operator and its input and returns a Future. Priority of operators
    to run and resources they are run on are deferred to specific OperatorScheduler
    implementations

    Base OperatorScheduler behaves as a simple queue deferring to ThreadPoolExecutor

    :param max_workers: maximum number of threads to execute at once
    """

    def __init__(self, max_workers: int = 1):
        self._threadpool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, *args, operator: Operator, **kwargs) -> Future:
        """
        :param operator: operator to run
        :param operator_input: input schema to the operator
        :param context: context of already run operators
        :return: future referencing the asynchronously run output of the operator
        """
        return self._threadpool.submit(operator, *args, **kwargs)

    def can_process(self, *args, operator: Operator, **kwargs) -> bool:
        """
        :param operator: operator to check
        :param operator_input: operator_input to check
        :return: True if this Operator can process the given operator and input.
            Base OperatorScheduler always returns True
        """
        return True
