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
from tests.deepsparse.transformers.pipelines.helpers import (
    ORTGroundTruthSource,
    TorchGroundTruthSource,
)


@pytest.mark.parametrize(
    "model_stub, model_name, uses_bos_token",
    [
        (
            "zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
            "huggingface/bigpython_bigquery_thepile/base-none",
            "salesforce/codegen-350m-mono",
            False,
        ),
    ],
    scope="class",
)
def test_ground_truth_sources(model_stub, model_name, uses_bos_token):
    num_tokens_generate = 256
    prompt = """
    Didn't know what time it was, the lights were low
    I leaned back on my radio
    Some cat was layin' down some rock 'n' roll
    "Lotta soul," he said
    Then the loud sound did seem to fade
    Came back like a slow voice on a wave of phase
    That weren't no DJ, that was hazy cosmic jive
    """

    torch_source = TorchGroundTruthSource(
        num_tokens_to_generate=num_tokens_generate, model_name=model_name
    )
    ort_source = ORTGroundTruthSource(
        num_tokens_to_generate=num_tokens_generate,
        model_name=model_name,
        model_stub=model_stub,
    )

    (
        torch_target_generated_logits,
        torch_target_prompt_logits,
        torch_target_prompt_cache,
    ) = torch_source(prompt)
    (
        ort_target_prompt_logits,
        ort_target_prompt_cache,
    ) = ort_source(prompt)

    # check that the prompt logits are the same
    assert numpy.allclose(
        torch_target_prompt_logits, ort_target_prompt_logits, atol=1e-4
    )
    # check that the prompt cache is the same
    for torch_cache, ort_cache in zip(
        torch_target_prompt_cache, ort_target_prompt_cache
    ):
        assert numpy.allclose(torch_cache, ort_cache, atol=1e-5)
