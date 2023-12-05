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

import time
from concurrent.futures import Future

import numpy

from deepsparse.v2.operators import EngineOperator
from deepsparse.v2.schedulers.utils import (
    ContinuousBatchingExecutorThread,
    ContinuousBatchingQueues,
)


def test_continuous_batching_executor_thread():
    # mobilenet model with batch_size=2
    engine_operator = EngineOperator("zoo:mobilenet_v2-1.0-imagenet-base")

    # create queues object and add operator
    queues = ContinuousBatchingQueues()
    queues.add_queue(engine_operator, batch_sizes=[1])

    # create engine map
    operators_to_engines = {engine_operator: {1: engine_operator.engine}}

    worker_thread = ContinuousBatchingExecutorThread(queues, operators_to_engines)

    # thread not started yet
    assert not worker_thread.is_alive()

    # start and assert thread is alive
    worker_thread.start()
    assert worker_thread.is_alive()

    # create first input and add it to queue
    input_1 = engine_operator.input_schema(
        engine_inputs=[numpy.random.randn(1, 3, 224, 224).astype(numpy.float32)]
    )
    future_1 = Future()
    queues.add_queue_item(engine_operator, input_1, future=future_1)

    # assert that future is not yet resolved
    assert not future_1.done()

    # create second input and add it to queue
    input_2 = engine_operator.input_schema(
        engine_inputs=[numpy.random.randn(1, 3, 224, 224).astype(numpy.float32)]
    )
    future_2 = Future()
    queues.add_queue_item(engine_operator, input_2, future=future_2)

    # wait 1 second to give engine time to complete
    time.sleep(1)

    assert future_1.done()
    assert future_2.done()

    result_1 = future_1.result()
    result_2 = future_2.result()

    assert isinstance(result_1, engine_operator.output_schema)
    assert isinstance(result_2, engine_operator.output_schema)

    def assert_batch_size_one(arrays):
        for array in arrays:
            assert array.shape[0] == 1

    # make sure only a single batch item was returned to each future
    # TODO: test that the correct bs1 item is returned (can test against bs1 engine)
    assert_batch_size_one(result_1.engine_outputs)
    assert_batch_size_one(result_2.engine_outputs)
