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

import numpy

from deepsparse.operators import EngineOperator
from deepsparse.schedulers import ContinuousBatchingScheduler


@pytest.mark.skip("debuging")
def test_continuous_batching_executor_thread():
    # simple test that ContinuousBatchingScheduler can be instantiated and return
    # a result from a request, for testing multi-batch execution, making enough
    # concurrent requests guarantee batched execution is out of scope
    scheduler = ContinuousBatchingScheduler()

    # mobilenet model with batch_size=2
    engine_operator = EngineOperator(
        "zoo:mobilenet_v2-1.0-imagenet-base",
    )

    scheduler.add_engine_operator(engine_operator, [2])

    # submit job to scheduler and expect future to be returned
    engine_input = engine_operator.input_schema(
        engine_inputs=[numpy.random.randn(1, 3, 224, 224).astype(numpy.float32)]
    )
    future = scheduler.submit(engine_input, operator=engine_operator)
    assert isinstance(future, Future)
    assert not future.done()  # assume this runs before engine has a chance to complete

    # assert that output resolves and contains a numpy array
    engine_output = future.result()
    assert isinstance(engine_output, engine_operator.output_schema)
    assert isinstance(engine_output.engine_outputs[0], numpy.ndarray)
