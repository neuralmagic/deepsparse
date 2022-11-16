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

import numpy

import pytest
from deepsparse import Pipeline


@pytest.fixture(scope="session")
def stub_and_task():
    yield (
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
        "imagenet/pruned_quant-moderate",
        "image_classification",
    )


# TODO: fix failing test
def test_embedding_extraction_pipeline(stub_and_task):
    stub, task = stub_and_task
    pipeline = Pipeline.create(
        task="embedding_extraction",
        model_path=stub,
        base_task=task,
    )

    inputs = numpy.random.random((3, 224, 224))
    embeddings = pipeline(inputs)
    assert embeddings
