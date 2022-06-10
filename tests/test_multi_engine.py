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

import pytest
import threading
from deepsparse import MultiModelEngine, Context
from deepsparse.utils import verify_outputs
from sparsezoo.models import classification
from sparsezoo.objects import Model


model_test_registry = {
    "mobilenet_v1": classification.mobilenet_v1,
    "mobilenet_v2": classification.mobilenet_v2,
    "resnet_18": classification.resnet_18,
    "efficientnet_b0": classification.efficientnet_b0,
}


@pytest.mark.parametrize(
    "num_streams, num_requests",
    (
        [
            pytest.param(num_streams, num_requests)
            for num_streams in [0, 2, 4]
            for num_requests in [1, 4, 8]
        ]
    ),
)
@pytest.mark.smoke
class TestMultiModelEngineParametrized:
    def thread_function(self, model: Model, batch_size: int, context: Context):
        m = model()
        batch = m.sample_batch(batch_size=batch_size)
        inputs = batch["inputs"]
        outputs = batch["outputs"]
        engine = MultiModelEngine(model=m, batch_size=batch_size, context=context)
        pred_outputs = engine(inputs)
        verify_outputs(pred_outputs, outputs)

    def test_multi_engine(
        self,
        num_streams: int,
        num_requests: int,
    ):
        models = list(model_test_registry.values())
        batch_size = 64
        context = Context(num_streams=num_streams)
        model_index = 0
        threads = list()

        for i in range(num_requests):
            thread = threading.Thread(
                target=self.thread_function,
                args=(models[model_index % len(models)], batch_size, context),
            )
            thread.start()
            threads.append(thread)
            ++model_index

        for thread in threads:
            thread.join()
