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

from typing import List, Optional, Tuple

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from tests.deepsparse.transformers.pipelines.helpers import (
    ORTGroundTruthSource,
    TorchGroundTruthSource,
)


@pytest.mark.parametrize(
    "use_deepsparse_cache",
    [True, False],
)
@pytest.mark.parametrize(
    "model_stub, model_name, uses_bos_token, logits_max_diff_kv_cache_has_been_filled",
    [
        (
            "zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
            "huggingface/bigpython_bigquery_thepile/base-none",
            "salesforce/codegen-350m-mono",
            False,
            15.5,
        ),
        # TODO: Waiting for the model to be available
        # ("zoo:nlg/text_generation/opt-1.3b/pytorch/huggingface/opt_pretrain/pruned50_quantW8A8-none",
        #  "facebook/opt-1.3b",
        #  True,
        #  None),
    ],
    scope="class",
)
class TestTextGenerationPipeline:
    """
    This test suite is meant to test the main scenarios of
    the text generation pipeline.
    """

    @pytest.fixture
    def setup(
        self,
        model_stub,
        model_name,
        uses_bos_token,
        logits_max_diff_kv_cache_has_been_filled,
        use_deepsparse_cache,
    ):
        self.num_tokens_generate = 216
        self.prompt = """
           Didn't know what time it was, the lights were low
           I leaned back on my radio
           Some cat was layin' down some rock 'n' roll
           "Lotta soul," he said
           Then the loud sound did seem to fade
           Came back like a slow voice on a wave of phase
           That weren't no DJ, that was hazy cosmic jive
           """
        # create torch ground source
        torch_source = TorchGroundTruthSource(
            num_tokens_to_generate=self.num_tokens_generate, model_name=model_name
        )
        torch_ground_truth = torch_source(self.prompt)

        # prompt length is expressed in number of prompt tokens
        prompt_length = torch_ground_truth[1].shape[1]

        # sequence_length that assures that the KV cache will not be filled up
        self.sequence_length = 2 * prompt_length + self.num_tokens_generate
        # sequence_length that assures that the KV cache will be filled up
        self.sequence_length_short = self.num_tokens_generate

        # prompt_processing_sequence_length used for the multitoken prefill scenario
        self.prompt_processing_sequence_length = 16

        # the maximum trheshold for the difference between the logits
        # when running a scenario where KV Cache buffer has been filled
        self.logits_max_diff_kv_cache_has_been_filled = (
            logits_max_diff_kv_cache_has_been_filled
        )
        self.use_deepsparse_cache = use_deepsparse_cache

        assert self.prompt_processing_sequence_length < prompt_length, (
            "The prompt processing sequence length "
            "must be smaller than the prompt length"
        )

        yield model_stub, model_name, uses_bos_token, torch_ground_truth

    def test_freeze_first_position(self, setup):
        # Test whether we should be "freezing" the first token after
        # the kv cache is full
        model_stub, _, uses_bos_token, _ = setup
        pipeline = Pipeline.create(task="text_generation", model_path=model_stub)
        assert pipeline.engine._freeze_first_position == uses_bos_token

    def test_ort_model(self, setup):
        # Assert that the ONNX model with KV Cache support runs
        # directly in ONNXRuntime and delivers correct results
        model_stub, model_name, _, torch_ground_truth = setup

        ort_source = ORTGroundTruthSource(
            model_name=model_name,
            model_stub=model_stub,
        )
        ort_prompt_logits, ort_prompt_kv_cache = ort_source(self.prompt)
        _, torch_prompt_logits, torch_prompt_cache, _ = torch_ground_truth

        # check that the prompt logits are the same
        assert numpy.allclose(torch_prompt_logits, ort_prompt_logits, atol=1e-4)
        # check that the prompt cache is the same
        for torch_cache, ort_cache in zip(torch_prompt_cache, ort_prompt_kv_cache):
            assert numpy.allclose(torch_cache, ort_cache, atol=1e-4)

    def test_ort_single_token_prefill(self, setup):
        # Test the pipeline that uses ORT engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by single-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed externally

        if self.use_deepsparse_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        model_stub, _, _, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length,
            prompt_processing_sequence_length=1,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
        )

    def test_ort_multi_token_prefill(self, setup):
        # Test the pipeline that uses ORT engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed externally

        if self.use_deepsparse_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        model_stub, _, _, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length,
            prompt_processing_sequence_length=self.prompt_processing_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
        )

    def test_ort_generation_after_kv_cache_has_been_filled(self, setup):
        # Test the pipeline that uses ORT engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is filled up (old entries are removed)
        # 3. KV Cache managed externally

        if self.use_deepsparse_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        model_stub, _, _, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_short,
            prompt_processing_sequence_length=self.prompt_processing_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens > self.sequence_length_short, (
            "for this scenario, the kv cache should be full: "
            "the total number of processed tokens should be "
            "greater than the sequence length"
        )

        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
            max_logits_difference_threshold=self.logits_max_diff_kv_cache_has_been_filled,  # noqa E501
        )

    def test_deepsparse_single_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by single-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed externally or internally

        model_stub, _, _, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length,
            prompt_processing_sequence_length=1,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            use_deepsparse_cache=self.use_deepsparse_cache,
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
            run_cache_validation=not self.use_deepsparse_cache,
        )

    def test_deepsparse_multi_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed externally or internally

        model_stub, _, _, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length,
            prompt_processing_sequence_length=self.prompt_processing_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            use_deepsparse_cache=self.use_deepsparse_cache,
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
            run_cache_validation=not self.use_deepsparse_cache,
        )

    def test_deepsparse_generation_after_kv_cache_has_been_filled(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is filled up (old entries are removed)
        # 3. KV Cache managed externally or internally

        model_stub, _, _, torch_ground_truth = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=self.sequence_length_short,
            prompt_processing_sequence_length=self.prompt_processing_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            use_deepsparse_cache=self.use_deepsparse_cache,
        )
        output = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        cache_session = pipeline.engine.kv_cache
        assert cache_session.total_num_processed_tokens > self.sequence_length_short, (
            "for this scenario, the kv cache should be full: "
            "the total number of processed tokens should be "
            "greater than the sequence length"
        )

        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=torch_ground_truth,
            run_cache_validation=not self.use_deepsparse_cache,
            max_logits_difference_threshold=self.logits_max_diff_kv_cache_has_been_filled,  # noqa E501
        )

    def test_run_same_prompt_multiple_times(self, setup):
        # Test the scenario, where the same prompt is run multiple times
        # Every run should produce the same output
        model_stub, *_ = setup
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            use_deepsparse_cache=self.use_deepsparse_cache,
        )
        output_1 = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        output_2 = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        assert output_1.sequences[0] == output_2.sequences[0]
        assert numpy.allclose(output_1.logits, output_2.logits, atol=1e-4)

    def _test_output(
        self,
        output: "TextGenerationOutput",  # noqa F821
        cache_session: DecoderKVCache,
        torch_ground_truth: Tuple[numpy.ndarray, ...],
        max_logits_difference_threshold: Optional[float] = None,
        run_cache_validation: bool = True,
    ):
        # extract numpy arrays from cached_inputs
        kv_cache_array = list(cache_session.cached_inputs.values())

        (
            generated_logits,
            prompt_logits,
            prompt_kv_cache,
            generated_text,
        ) = torch_ground_truth

        # concatenate target prompt_logits and generated_logits and check
        target_logits = numpy.concatenate([prompt_logits, generated_logits], axis=1)

        if max_logits_difference_threshold:
            # if comparing the output from the model where
            # the kv cache has been filled, we expect the
            # maximum absolute difference between the logits
            # to be less than the threshold
            # (the threshold is established by running the
            # ONNX model in ONNXRuntime)
            assert (
                abs(output.logits - target_logits).max()
                < max_logits_difference_threshold
            )
        else:
            # otherwise, we expect the logits to be exactly the same
            # as the target logits; the generated sequence should
            # also be the same as the target sequence, and finally
            # (if applicable) the kv cache should be the same as the
            # target kv cache
            assert numpy.allclose(output.logits, target_logits, atol=1e-4)
            assert self.prompt + output.sequences[0] == generated_text

            if run_cache_validation:
                self._test_kv_cache_state(
                    expected_cache=kv_cache_array,
                    target_cache=torch_ground_truth[2],
                    total_num_processed_tokens=cache_session.total_num_processed_tokens,
                )

    @staticmethod
    def _test_kv_cache_state(
        expected_cache: List[numpy.ndarray],
        target_cache: List[numpy.ndarray],
        total_num_processed_tokens: int,
    ):
        for x, y in zip(expected_cache, target_cache):
            start_index = total_num_processed_tokens
            end_index = total_num_processed_tokens - y.shape[2]
            # x is (in general) composed of three arrays:
            # - padding cache entries (from 0 to -start_index)
            # - prompt cache entries (from -start_index to -end_index)
            # - generated cache entries (from -end_index to -1)
            # as target_cache only pertains to prompt cache entries, we need to
            # compare only the prompt cache entries in x with y
            assert numpy.allclose(x[:, :, -start_index:-end_index, :], y, atol=1e-4)
