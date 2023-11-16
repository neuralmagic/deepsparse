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

from threading import Thread
from typing import Dict

from deepsparse import Engine
from deepsparse.v2.operators import EngineOperator
from deepsparse.v2.schedulers.utils.continuous_batching_queues import (
    ContinuousBatchingQueues,
)


__all__ = [
    "ContinuousBatchingExecutorThread",
]


class ContinuousBatchingExecutorThread(Thread):
    """
    Thread that when started runs indefinitely, grabbing a valid batch from
    the queues when possible and running them in the correct engine

    :param queues: ContinuousBatchingQueues object containing a queue for
        each valid engine
    :param operators_to_engines: dictionary mapping valid engine operators
        to a dictionary of its valid batch sizes mapped to an engine compiled
        for that batch size
    """

    def __init__(
        self,
        queues: ContinuousBatchingQueues,
        operators_to_engines: Dict[EngineOperator, Dict[int, Engine]],
    ):
        self._queues = queues
        self._operators_to_engines = operators_to_engines
        self._should_stop = False

        super().__init__(target=self._working_loop)
        self.daemon = True  # worker thread should exit when main thread exits

    def _working_loop(self):
        # indefinitely wait for batch, run batch, split and resolve futures
        while True:
            # wait for next batch to be available
            engine_operator, batch = self._queues.pop_batch(block=True)

            # unpack batch of QueueEntry objects
            engine_inputs, futures, _ = list(zip(*batch))
            batch_size = len(engine_inputs)

            # type is EngineOperatorInputs
            joined_inputs = engine_operator.input_schema.join(engine_inputs)

            # get engine for this operator compiled to the popped batch size
            # and set the inputs to execute with it
            joined_inputs.engine = self._operators_to_engines[engine_operator][
                batch_size
            ]

            # run the engine operator with the given engine at the joined batch size
            joined_outputs = engine_operator(joined_inputs, inference_state=None)

            # split outputs and return the results to their respective futures
            split_outputs = joined_outputs.split()
            for output, future in zip(split_outputs, futures):
                future.set_result(output)
