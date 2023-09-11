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
    CODE_LANGUAGE_PROMPT,
    ORTGroundTruthSource,
    TorchGroundTruthSource,
    generate_pytest_params,
)


# ------ Tests configuration ------
# The following parameters are used to configure the tests

# STUBS_TO_TEST should be a list of tuples in the form:
# [(model_stub, model_name, prompt), ...], where model_stub is
# the stub of the sparsezoo model to be tested, model_name
# is the name of the HuggingFace model that corresponds to the
# sparsezoo model and prompt is the prompt that will be used
# for the text generation pipeline. The prompt should be
# representative of the domain that the model was trained on.

# The overarching goal of this suite is to
# benchmark the sparsezoo models against the HuggingFace models
# and this is why we need to provide both sources of truth
STUBS_TO_TEST = [
    (
        "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",  # noqa E501
        "salesforce/codegen-350m-mono",
        CODE_LANGUAGE_PROMPT,
    ),
    # TODO: Waiting for the model to become available
    # ("zoo:nlg/text_generation/opt-1.3b/pytorch/huggingface/opt_pretrain/base-none",
    # "facebook/opt-1.3b",
    # NATURAL_LANGUAGE_PROMPT),
]

# RUN_BASE_TESTS_ONLY is a boolean flag that is used to run only
# the essential test that cover the hot path of the text generation pipeline.
# If set to False, the test suite will run all the tests that are available.
# Note, that if this parameter is set to True, there are some tests that will
# be skipped if the cache (see the test suite for more details)
RUN_BASE_TESTS_ONLY = False

# CACHE_MANAGEMENT_TYPE should be a list of strings that can be either
# "internal" or/and "external", but cannot be empty. This parameter
# is used to test the different cache management strategies that
# are available for the text generation pipeline. Note, that there
# are some tests that will be skipped if the cache management type
# is set to "internal" only (see the test suite for more details)
CACHE_MANAGEMENT_TYPE = ["internal", "external"]

# LOGITS_THRESHOLD is an optional config (could be set to an empty list),
# that is used to test the difference between the actual and the expected
# logits in the situations where we will not be able to match the ground
# truth (e.g. when the DeepSparse pipeline is running after the KV cache
# has been filled up). This value is established empirically for each
# combination of prompt/pipeline/num_generated tokens. To enable full testing,
# set this list so that it has the same length as the STUBS_TO_TEST list.
# To disable it, set it to an empty list (e.g. LOGITS_THRESHOLDS = []).
# The testing suite will notify the user about the tests that will be
# skipped as a result
LOGITS_THRESHOLDS = [13.0]
# ----------------


pytest_params = generate_pytest_params(
    STUBS_TO_TEST, CACHE_MANAGEMENT_TYPE, LOGITS_THRESHOLDS
)


@pytest.mark.parametrize(
    "internal_kv_cache",
    pytest_params[0],
)
@pytest.mark.parametrize(
    "model_stub, model_name, prompt, uses_bos_token, logits_threshold",
    pytest_params[1],
    scope="class",
)
class TestTextGenerationPipeline:
    """
    This test suite is meant to test the main scenarios of
    the text generation pipeline.
    """

    def get_pipeline(self, **kwargs):
        if not kwargs:
            # return the default pipeline
            if self.default_pipeline:
                return self.default_pipeline
            else:
                self.default_pipeline = Pipeline.create(
                    task="text_generation",
                    model_path=self.model_stub,
                    internal_kv_cache=self.internal_kv_cache,
                    prompt_sequence_length=self.prompt_sequence_length,
                    sequence_length=self.sequence_length,
                    max_generated_tokens=self.num_tokens_generate,
                    force_max_tokens=True,
                )
                return self.default_pipeline
        # return a pipeline with the given kwargs
        return Pipeline.create(**kwargs)

    @pytest.fixture
    def setup(
        self,
        model_stub,
        model_name,
        prompt,
        uses_bos_token,
        logits_threshold,
        internal_kv_cache,
    ):
        self.num_tokens_generate = 216
        self.model_stub = model_stub
        self.prompt = prompt
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

        # prompt_sequence_length used for the multitoken prefill scenario
        self.prompt_sequence_length = prompt_length // 2

        # the maximum threshold for the difference between the logits
        # when running a scenario where KV Cache buffer has been filled
        self.logits_threshold = logits_threshold

        self.internal_kv_cache = internal_kv_cache

        self.default_pipeline = None

        assert self.prompt_sequence_length < prompt_length, (
            "The prompt processing sequence length "
            "must be smaller than the prompt length"
        )

        yield model_name, uses_bos_token, torch_ground_truth

    @pytest.mark.skipif(
        RUN_BASE_TESTS_ONLY,
        reason="RUN_BASE_TESTS_ONLY = True, running only essential tests as a result",
    )
    def test_freeze_first_position(self, setup):
        # Test whether we should be "freezing" the first token after
        # the kv cache is full
        _, uses_bos_token, _ = setup
        pipeline = self.get_pipeline()
        assert pipeline.engine._freeze_first_position == uses_bos_token

    @pytest.mark.skipif(
        RUN_BASE_TESTS_ONLY,
        reason="RUN_BASE_TESTS_ONLY = True, running only essential tests as a result",
    )
    def test_ort_model(self, setup):
        # Assert that the ONNX model with KV Cache support runs
        # directly in ONNXRuntime and delivers correct results
        model_name, _, torch_ground_truth = setup

        ort_source = ORTGroundTruthSource(
            model_name=model_name,
            model_stub=self.model_stub,
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

        if self.internal_kv_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        _, _, torch_ground_truth = setup
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            sequence_length=self.sequence_length,
            prompt_sequence_length=1,
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

        if self.internal_kv_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        _, _, torch_ground_truth = setup
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            sequence_length=self.sequence_length,
            prompt_sequence_length=self.prompt_sequence_length,
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

        if self.internal_kv_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        _, _, torch_ground_truth = setup
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            sequence_length=self.sequence_length_short,
            prompt_sequence_length=self.prompt_sequence_length,
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
            run_logits_difference_validation=True,
            logits_threshold=self.logits_threshold,
        )

    def test_deepsparse_single_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by single-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed externally or internally

        _, _, torch_ground_truth = setup
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            sequence_length=self.sequence_length,
            prompt_sequence_length=1,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            internal_kv_cache=self.internal_kv_cache,
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
            run_cache_validation=not self.internal_kv_cache,
        )

    def test_deepsparse_multi_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed externally or internally

        _, _, torch_ground_truth = setup
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            sequence_length=self.sequence_length,
            prompt_sequence_length=self.prompt_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            internal_kv_cache=self.internal_kv_cache,
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
            run_cache_validation=not self.internal_kv_cache,
        )

    def test_deepsparse_generation_after_kv_cache_has_been_filled(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is filled up (old entries are removed)
        # 3. KV Cache managed externally or internally

        _, _, torch_ground_truth = setup
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            sequence_length=self.sequence_length_short,
            prompt_sequence_length=self.prompt_sequence_length,
            max_generated_tokens=self.num_tokens_generate,
            force_max_tokens=True,
            internal_kv_cache=self.internal_kv_cache,
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
            run_cache_validation=not self.internal_kv_cache,
            run_logits_difference_validation=True,
            logits_threshold=self.logits_threshold,
        )

    @pytest.mark.skipif(
        RUN_BASE_TESTS_ONLY,
        reason="RUN_BASE_TESTS_ONLY = True, running only essential tests as a result",
    )
    def test_run_same_prompt_multiple_times(self, setup):
        # Test the scenario, where the same prompt is run multiple times
        # Every run should produce the same output
        pipeline = self.get_pipeline()

        output_1 = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        output_2 = pipeline(
            sequences=self.prompt, return_logits=True, include_prompt_logits=True
        )
        assert output_1.sequences[0] == output_2.sequences[0]
        assert numpy.allclose(output_1.logits, output_2.logits, atol=1e-4)

    @pytest.mark.skipif(
        RUN_BASE_TESTS_ONLY,
        reason="RUN_BASE_TESTS_ONLY = True, running only essential tests as a result",
    )
    def test_run_multiple_prompts_in_parallel(self, setup):
        # Test the scenario, where multiple prompts are run in parallel
        # Same two prompts should produce the same output
        pipeline = self.get_pipeline()

        output = pipeline(
            sequences=[self.prompt, self.prompt],
            return_logits=True,
            include_prompt_logits=True,
        )

        assert numpy.allclose(output.logits[0], output.logits[1], atol=1e-4)
        assert output.sequences[0] == output.sequences[1]

    def _test_output(
        self,
        output: "TextGenerationOutput",  # noqa F821
        cache_session: DecoderKVCache,
        torch_ground_truth: Tuple[numpy.ndarray, ...],
        logits_threshold: Optional[float] = None,
        run_logits_difference_validation: bool = False,
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

        if run_logits_difference_validation:
            # if comparing the output from the model where
            # the kv cache has been filled, we expect the
            # maximum absolute difference between the logits
            # to be less than the threshold
            # (the threshold is established by running the
            # ONNX model in ONNXRuntime)
            if logits_threshold is None:
                pytest.skip(
                    "Attempting to run logits difference "
                    "validation, but logits_threshold is None"
                )
            assert abs(output.logits - target_logits).max() < logits_threshold
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
