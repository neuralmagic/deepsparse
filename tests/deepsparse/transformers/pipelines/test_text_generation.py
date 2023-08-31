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

from typing import List, Tuple

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from tests.deepsparse.transformers.pipelines.helpers import (
    ORTGroundTruthSource,
    TorchGroundTruthSource,
)


# using ORT as the engine
# running with single token prefill
# not running past the max capacity of the KV cache
# (ensured by setting the sequence length to be twice
# the number of tokens to generate) TODO make it variable
@pytest.mark.parametrize(
    "model_stub, model_name, uses_bos_token",
    [
        # (
        #     "zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
        #     "huggingface/bigpython_bigquery_thepile/base-none",
        #     "salesforce/codegen-350m-mono",
        #     False,
        # ),
        ("/home/ubuntu/damian/sparseml/deployment_opt", "facebook/opt-350m", True)
    ],
    scope="class",
)
class TestTextGenerationPipeline:
    @pytest.fixture
    def setup(self, model_stub, model_name, uses_bos_token):

        self.num_tokens_generate = 64
        self.prompt = """
           Didn't know what time it was, the lights were low
           I leaned back on my radio
           Some cat was layin' down some rock 'n' roll
           "Lotta soul," he said
           Then the loud sound did seem to fade
           Came back like a slow voice on a wave of phase
           That weren't no DJ, that was hazy cosmic jive
           """
        torch_source = TorchGroundTruthSource(
            num_tokens_to_generate=self.num_tokens_generate, model_name=model_name
        )

        torch_ground_truth = torch_source(self.prompt)
        prompt_length = torch_ground_truth[1].shape[
            1
        ]  # expressed in number of prompt tokens

        # sequence_length that assures that the KV cache will not be filled up
        self.sequence_length_large = (
            prompt_length + self.num_tokens_generate + prompt_length
        )
        # sequence_length that assures that the KV cache will be filled up
        self.sequence_length_short = (
            prompt_length + self.num_tokens_generate - prompt_length
        )
        # prompt_processing_sequence_length used for the multitoken prefill scenario
        self.prompt_processing_sequence_length = 16

        assert self.prompt_processing_sequence_length < prompt_length

        yield model_stub, model_name, uses_bos_token, torch_ground_truth

    def test_freeze_first_position(self, setup):
        # test whether we should be "freezing" the first token after
        # the kv cache is full
        model_stub, _, uses_bos_token, _ = setup
        pipeline = Pipeline.create(task="text_generation", model_path=model_stub)
        assert pipeline.engine._freeze_first_position == uses_bos_token

    def test_ort_model(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        ort_source = ORTGroundTruthSource(
            model_name=model_name,
            model_stub=model_stub,
        )
        ort_prompt_logits, ort_prompt_kv_cache = ort_source(self.prompt)
        _, torch_target_prompt_logits, torch_target_prompt_cache = torch_ground_truth
        # check that the prompt logits are the same
        assert numpy.allclose(torch_target_prompt_logits, ort_prompt_logits, atol=1e-4)
        # check that the prompt cache is the same
        for torch_cache, ort_cache in zip(
            torch_target_prompt_cache, ort_prompt_kv_cache
        ):
            assert numpy.allclose(torch_cache, ort_cache, atol=1e-5)

    def test_ort_single_token_prefill_kv_cache_not_full(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_large,
            prompt_processing_sequence_length=1,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length_large
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
        )

    def test_ort_multi_token_prefill_kv_cache_not_full(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_large,
            prompt_processing_sequence_length=self.prompt_processing_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length_large
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
        )

    def test_ort_single_token_prefill_kv_cache_full(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_short,
            prompt_processing_sequence_length=1,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens > self.sequence_length_short
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
        )

    def test_deepsparse_single_token_prefill_kv_cache_not_full_external(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_large,
            prompt_processing_sequence_length=1,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            use_deepsparse_cache=False,
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length_large
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
        )

    def test_deepsparse_multi_token_prefill_kv_cache_not_full_external(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_large,
            prompt_processing_sequence_length=self.prompt_processing_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            use_deepsparse_cache=False,
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length_large
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
        )

    def test_deepsparse_single_token_prefill_kv_cache_not_full_internal(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_large,
            prompt_processing_sequence_length=1,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < 256
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
            run_cache_validation=False,
        )

    def test_deepsparse_multi_token_prefill_kv_cache_not_full_internal(self, setup):
        model_stub, model_name, uses_bos_token, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_large,
            prompt_processing_sequence_length=self.prompt_processing_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length_large
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
            run_cache_validation=False,
        )

    def _test_output(
        self,
        output: "TextGenerationOutput",
        cache_session: DecoderKVCache,
        torch_ground_truth: Tuple[numpy.ndarray, ...],
        run_cache_validation: bool = True,
    ):
        # extract numpy arrays from cached_inputs
        kv_cache_array = list(cache_session.cached_inputs.values())
        generated_logits, prompt_logits, prompt_kv_cache = torch_ground_truth
        # concatenate target prompt_logits and generated_logits and check
        # that they match the output
        target_logits = numpy.concatenate([prompt_logits, generated_logits], axis=1)
        assert numpy.allclose(output.logits, target_logits, atol=1e-4)
        # run cache validation if requested
        if run_cache_validation:
            self._test_kv_cache_state(
                expected_cache=kv_cache_array,
                target_cache=torch_ground_truth[2],
                total_num_processed_tokens=cache_session.total_num_processed_tokens,
            )

    def _test_kv_cache_state(
        self,
        expected_cache: List[numpy.ndarray],
        target_cache: List[numpy.ndarray],
        total_num_processed_tokens: int,
    ):
        for x, y in zip(expected_cache, target_cache):
            start_index = total_num_processed_tokens
            end_index = total_num_processed_tokens - y.shape[2]
            # x is in general composed of three arrays:
            # - padding cache entries (from 0 to -start_index)
            # - prompt cache entries (from -start_index to -end_index)
            # - generated cache entries (from -end_index to -1)
            # as target_cache only contains prompt cache entries, we need to
            # compare only the prompt cache entries in x with y
            assert numpy.allclose(x[:, :, -start_index:-end_index, :], y, atol=1e-4)
