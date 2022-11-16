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


@pytest.mark.parametrize(
    "model_path,task,input_kwargs_lambda",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
            "imagenet/pruned_quant-moderate",
            "image_classification",
            lambda: dict(images=numpy.random.random((1, 3, 224, 224))),
        ),
    ],
)
@pytest.mark.parametrize(
    "emb_extraction_layer",
    [1, None],
)
def test_embedding_extraction_pipeline(
    model_path, task, input_kwargs_lambda, emb_extraction_layer
):
    pipeline = Pipeline.create(
        task="embedding_extraction",
        model_path=model_path,
        base_task=task,
        emb_extraction_layer=-1,
    )

    input_kwargs = input_kwargs_lambda()
    outputs = pipeline(**input_kwargs)
    assert outputs
    assert len(outputs.embeddings) > 0
    assert isinstance(outputs.embeddings[0], numpy.ndarray)
