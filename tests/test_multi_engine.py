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

import pytest
from deepsparse import Context, MultiModelEngine
from deepsparse.utils import verify_outputs
from sparsezoo import Model


model_test_registry = {
    "mobilenet_v1": (
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"
    ),
    "mobilenet_v2": (
        "zoo:cv/classification/mobilenet_v2-1.0/pytorch/sparseml/imagenet/base-none"
    ),
    "resnet_18": (
        "zoo:cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/base-none"
    ),
    "efficientnet_b0": (
        "zoo:cv/classification/efficientnet-b0/pytorch/sparseml/imagenet/base-none"
    ),
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
        m = Model(model)
        inputs, outputs = _get_sample_inputs_outputs(m, batch_size)
        engine = MultiModelEngine(model=m, batch_size=batch_size, context=context)
        pred_outputs = engine(inputs)
        verify_outputs(pred_outputs, outputs)

    def test_multi_engine(
        self,
        num_streams: int,
        num_requests: int,
    ):
        models = list(model_test_registry.values())
        batch_size = 1
        context = Context(num_streams=num_streams)
        model_index = 0
        threads = list()

        for i in range(num_requests):
            thread = Thread(
                target=self.thread_function,
                args=(models[model_index % len(models)], batch_size, context),
            )
            thread.start()
            threads.append(thread)
            model_index += 1

        for thread in threads:
            thread.join()


def _get_sample_inputs_outputs(model: Model, batch_size: int):
    batch = model.sample_batch(batch_size=batch_size)

    input_key = next(key for key in batch.keys() if "input" in key)
    output_key = next(key for key in batch.keys() if "output" in key)

    return batch[input_key], batch[output_key]
