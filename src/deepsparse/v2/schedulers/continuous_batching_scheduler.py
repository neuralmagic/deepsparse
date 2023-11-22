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
from threading import Lock
from typing import List

from deepsparse.v2.operators import EngineOperator, Operator
from deepsparse.v2.schedulers.scheduler import OperatorScheduler
from deepsparse.v2.schedulers.utils import (
    ContinuousBatchingExecutorThread,
    ContinuousBatchingQueues,
)


__all__ = ["ContinuousBatchingScheduler"]


_GLOBAL_SCHEDULER = None


class ContinuousBatchingScheduler(OperatorScheduler):
    """
    Manages EngineOperator jobs that should be run with continuous batching.
    Groups requests for the same engine into larger batches and returns
    the result to the respective request threads after scheduled completion

    Example code for getting or creating a shared instance for scheduling
    between pipelines and adding an engine operator to the scheduler
    within a pipeline

    ```python

    class MyPipeline(Pipeline):

        def __init__(self):
            ...
            engine_operator = EngineOperator(...)
            ...
            continuous_batching_scheduler = ContinuousBatchingScheduler.get_instance()
            continuous_batching_scheduler.add_engine_operator(engine_operator, [1])

            super.__init__(...)
    ```

    :param max_workers: maximum number of threads to execute at once, default 1
    """

    # TODO: If the singleton always returns max_workers 1, should we remove this arg/not
    # give the user a choice?
    def __init__(self, max_workers: int = 1):
        self._max_workers = max_workers

        self._mutex = Lock()

        # Dict[EngineOperator, Dict[batch_size, Engine]]
        self._operators_to_engines = {}  # EngineOperator -> Dict[batch_size, Engine]
        self._queues = ContinuousBatchingQueues()

        # create and start max number of worker threads
        self._threads = [
            ContinuousBatchingExecutorThread(self._queues, self._operators_to_engines)
            for _ in range(self.max_workers)
        ]
        for worker_thread in self._threads:
            worker_thread.start()

    @classmethod
    def get_instance(cls) -> "ContinuousBatchingScheduler":
        """
        :return: global instance of the continuous batching scheduler. If one
            does not exist yet, a scheduler with a single worker thread to
            schedule all jobs is created and started
        """
        global _GLOBAL_SCHEDULER

        if _GLOBAL_SCHEDULER is not None:
            return _GLOBAL_SCHEDULER  # noqa: F823

        _GLOBAL_SCHEDULER = cls(max_workers=1)
        return _GLOBAL_SCHEDULER

    @property
    def max_workers(self) -> int:
        """
        :return: maximum number of threads to execute at once
        """
        return self._max_workers

    def submit(self, *args, operator: Operator, **kwargs) -> Future:
        """
        :param operator: operator to run
        :param operator_input: input schema to the operator
        :return: future referencing the asynchronously run output of the operator
        """
        inputs = args[0]
        if not isinstance(inputs, operator.input_schema):
            raise ValueError(
                "Inputs to ContinuousBatchingScheduler must be the specific "
                f"input schema to the given operator. Expected {operator.input_schema}"
                f"found {type(inputs)}"
            )

        future = Future()
        self._queues.add_queue_item(key=operator, item=inputs, future=future)

        return future

    def can_process(self, *args, operator: Operator, **kwargs) -> bool:
        """
        :param operator: operator to check
        :param operator_input: operator_input to check
        :return: True if this Operator can process the given operator and input.
            SchedulerGroup always returns True
        """
        return operator in self._operators_to_engines and operator in self._queues

    def add_engine_operator(
        self, engine_operator: EngineOperator, batch_sizes: List[int]
    ):
        """
        Adds tracking for an engine operator to this scheduler
        with continuous batching for the given sizes

        :param engine_operator: an EngineOperator, must be compiled with
            batch_size=1
        :param batch_sizes: batch sizes to use for continuous batching
        """
        # lock updates to _operators_to_engines while updating
        self._mutex.acquire()

        # validation
        if engine_operator in self._operators_to_engines:
            # operator already added
            return

        if not isinstance(engine_operator, EngineOperator):
            raise ValueError(
                f"Expected an EngineOperator instance, found {type(engine_operator)}"
            )
        if engine_operator.batch_size != 1:
            raise ValueError(
                "For continuous batching, EngineOperator must have batch_size=1. "
                f"found batch_size={engine_operator.batch_size}"
            )

        # build EngineOperator -> List[batch_size] dict
        operator_engines = {}
        # base engine, expected batch size is 1
        operator_engines[engine_operator.batch_size] = engine_operator.engine

        # compile auxillary engines for continuous batching
        for batch_size in batch_sizes:
            if batch_size == 1:
                continue  # already added

            override_model_path = None
            # text generation/NLEngineOperator specific; could add generic method
            # for all engine_operators, if desired
            if hasattr(engine_operator, "override_model_inputs"):
                override_model_path = engine_operator.override_model_inputs(
                    model_path=engine_operator.model_path, batch_size=batch_size
                )

            # will break for internal kv_cache; needs additional argument
            operator_engines[batch_size] = engine_operator.create_engine(
                batch_size=batch_size, model_path=override_model_path
            )

        self._operators_to_engines[engine_operator] = operator_engines
        self._queues.add_queue(
            key=engine_operator,
            batch_sizes=list(operator_engines.keys()),
        )

        # release lock
        self._mutex.release()
