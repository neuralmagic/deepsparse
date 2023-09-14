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

import os
from typing import List, Optional, Tuple, Union

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from tests.deepsparse.transformers.pipelines.helpers import (
    TorchGroundTruthSource,
    generate_pytest_params,
)


# --------- CONFIG ---------
# The following constants are used to configure the test suite

# PRECISION defines the threshold for comparing ground truth
# and pipeline output values
PRECISION: float = 1e-3
# RUN_TEST_ON_SPARSEZOO_MODELS defines whether the test suite
# should run on all the models available in SparseZoo or only
# a lightweight model that tests the overall functionality of
# the pipeline
RUN_TEST_ON_SPARSEZOO_MODELS: bool = os.environ.get("FULL_LLM_TESTING", False)
# CACHE_MANAGEMENT_TYPE defines the type of cache management
# that should be used by the pipeline. The following options
# are available: "external" and "internal"
CACHE_MANAGEMENT_TYPE: List[str] = ["internal", "external"]
# ----------------


pytest_params = generate_pytest_params(
    RUN_TEST_ON_SPARSEZOO_MODELS, CACHE_MANAGEMENT_TYPE
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
        # if no kwargs provided, returns the cached "default"
        # pipeline that is used for most of the tests.
        # Otherwise, returns a pipeline with the given kwargs
        # (the default pipeline kwargs are updated with the
        # user-provided kwargs)
        if not kwargs:
            # return the default pipeline
            if self.default_pipeline is None:
                self.default_pipeline = Pipeline.create(**self.default_pipeline_kwargs)
            return self.default_pipeline
        # return a pipeline with the given kwargs
        # update the default pipeline kwargs
        updated_kwargs = self.default_pipeline_kwargs.copy()
        updated_kwargs.update(kwargs)
        return Pipeline.create(**updated_kwargs)

    def run_pipeline(
        self,
        pipeline,
        sequences: Union[None, str, List[str]] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        # helper function that runs the pipeline
        return pipeline(
            sequences=sequences or self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=max_tokens or self.num_tokens_generate,
            **kwargs,
        )

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
        self.num_tokens_generate = 128
        self.model_stub = model_stub
        self.prompt = prompt
        self.uses_bos_token = uses_bos_token
        self.model_name = model_name

        # create torch ground source
        torch_source = TorchGroundTruthSource(
            num_tokens_to_generate=self.num_tokens_generate, model_name=model_name
        )
        self.torch_ground_truth = torch_source(self.prompt)

        # prompt length is expressed in number of prompt tokens
        prompt_length = self.torch_ground_truth[1].shape[1]

        # sequence_length that assures that the KV cache will not be filled up
        self.sequence_length = 2 * prompt_length + self.num_tokens_generate
        # sequence_length that assures that the KV cache will be filled up
        self.sequence_length_short = self.num_tokens_generate

        # prompt_sequence_length used for the multitoken prefill scenario,
        self.prompt_sequence_length = prompt_length // 2

        # the maximum threshold for the difference between the logits
        # when running a scenario where KV Cache buffer has been filled
        self.logits_threshold = logits_threshold
        self.internal_kv_cache = internal_kv_cache
        self.default_pipeline_kwargs = dict(
            task="text_generation",
            model_path=self.model_stub,
            internal_kv_cache=self.internal_kv_cache,
            prompt_sequence_length=self.prompt_sequence_length,
            sequence_length=self.sequence_length,
            force_max_tokens=True,
        )
        self.default_pipeline = None

        assert self.prompt_sequence_length < prompt_length, (
            "The prompt processing sequence length "
            "must be smaller than the prompt length"
        )

    def test_freeze_first_position(self, setup):
        # Test whether we should be "freezing" the first token after
        # the kv cache is full
        pipeline = self.get_pipeline()
        assert pipeline.engine.freeze_first_position == self.uses_bos_token

    def test_ort_single_token_prefill(self, setup):
        # Test the pipeline that uses ORT engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by single-token engine
        # 2. The KV Cache is never filled up
        if self.internal_kv_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        pipeline = self.get_pipeline(
            prompt_sequence_length=1, engine_type="onnxruntime"
        )
        output = self.run_pipeline(pipeline)
        cache_session = self._pop_cache_session(pipeline)
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=self.torch_ground_truth,
        )

    def test_ort_multi_token_prefill(self, setup):
        # Test the pipeline that uses ORT engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is never filled up
        if self.internal_kv_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        pipeline = self.get_pipeline(
            prompt_sequence_length=self.prompt_sequence_length,
            engine_type="onnxruntime",
        )
        output = self.run_pipeline(pipeline)
        cache_session = self._pop_cache_session(pipeline)
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=self.torch_ground_truth,
        )

    def test_ort_generation_after_kv_cache_has_been_filled(self, setup):
        # Test the pipeline that uses ORT engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is filled up (old entries are removed)

        if self.internal_kv_cache:
            pytest.skip(
                "Cannot run ORT pipeline with the internal deepsparse cache enabled."
            )
        pipeline = self.get_pipeline(
            sequence_length=self.sequence_length_short, engine_type="onnxruntime"
        )
        output = self.run_pipeline(pipeline)
        cache_session = self._pop_cache_session(pipeline)
        assert cache_session.total_num_processed_tokens > self.sequence_length_short, (
            "for this scenario, the kv cache should be full: "
            "the total number of processed tokens should be "
            "greater than the sequence length"
        )
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=self.torch_ground_truth,
            logits_threshold=self.logits_threshold,
        )

    def test_deepsparse_single_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by single-token engine
        # 2. The KV Cache is never filled up

        pipeline = self.get_pipeline(
            prompt_sequence_length=1,
            internal_kv_cache=self.internal_kv_cache,
        )
        output = self.run_pipeline(pipeline)
        cache_session = self._pop_cache_session(pipeline)
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=self.torch_ground_truth,
            run_cache_validation=not self.internal_kv_cache,
        )

    def test_deepsparse_multi_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is never filled up
        pipeline = self.get_pipeline()
        output = self.run_pipeline(pipeline)
        cache_session = self._pop_cache_session(pipeline)
        assert cache_session.total_num_processed_tokens < self.sequence_length
        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=self.torch_ground_truth,
            run_cache_validation=not self.internal_kv_cache,
        )

    def test_deepsparse_generation_after_kv_cache_has_been_filled(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is filled up (old entries are removed)
        # 3. KV Cache managed externally or internally

        pipeline = self.get_pipeline(
            sequence_length=self.sequence_length_short,
        )
        output = self.run_pipeline(pipeline)
        cache_session = self._pop_cache_session(pipeline)
        assert cache_session.total_num_processed_tokens > self.sequence_length_short, (
            "for this scenario, the kv cache should be full: "
            "the total number of processed tokens should be "
            "greater than the sequence length"
        )

        self._test_output(
            output=output,
            cache_session=cache_session,
            torch_ground_truth=self.torch_ground_truth,
            run_cache_validation=not self.internal_kv_cache,
            logits_threshold=self.logits_threshold,
        )

    def test_run_same_prompt_multiple_times(self, setup):
        # Test the scenario, where the same prompt is run multiple times
        # Every run should produce the same output
        pipeline = self.get_pipeline()

        output_1 = self.run_pipeline(pipeline)
        output_2 = self.run_pipeline(pipeline)

        assert output_1.sequences[0] == output_2.sequences[0]
        assert numpy.allclose(output_1.logits, output_2.logits, atol=PRECISION)

    def test_run_multiple_prompts_in_parallel(self, setup):
        # Test the scenario, where multiple prompts are run in parallel
        # Same two prompts should produce the same output
        pipeline = self.get_pipeline()

        output = self.run_pipeline(pipeline, sequences=[self.prompt, self.prompt])

        assert numpy.allclose(output.logits[0], output.logits[1], atol=PRECISION)
        assert output.sequences[0] == output.sequences[1]

    def test_num_generated_predictions(self, setup):
        # Test the scenario, where multiple predictions are generated
        # from the same prompt
        pipeline = self.get_pipeline()

        output_sequences = self.run_pipeline(pipeline, num_generated_predictions=2)
        assert len(output_sequences.sequences[0]) == 2
        output_sequences = self.run_pipeline(
            pipeline, sequences=[self.prompt, self.prompt], num_generated_predictions=2
        )
        assert len(output_sequences.sequences) == 2
        for sequences in output_sequences.sequences:
            assert len(sequences) == 2

    def test_run_with_same_session_ids(self, setup):
        # Test the scenario where the same session ids are used for multiple
        # inference runs. There are two conditions that must be fulfilled:
        # 1. The information regarding the prompt does not leak between sessions
        # 2. Running two prompts one after another is identical to running
        #    a composition of those prompts i.e.
        #     generated_text = pipeline(prompt_1)
        #     generated_text_2 = pipeline(prompt_2)
        #     generated_text_2 == pipeline(prompt_1 + generated_text + prompt_2)

        prompt_1 = "This prompt is used for testing purposes. To this to make sure that"
        prompt_2 = " still this prompt should not"
        num_generated_tokens = 32
        for multi_token_prefill in [True, False]:
            self._test_run_with_same_session_ids(
                prompt_1, prompt_2, num_generated_tokens, multi_token_prefill
            )

    def _test_run_with_same_session_ids(
        self,
        prompt_1,
        prompt_2,
        num_generated_tokens,
        multi_token_prefill,
    ):
        pipeline = self.get_pipeline(
            prompt_sequence_length=self.prompt_sequence_length
            if multi_token_prefill
            else 1,
        )
        for session_id_pair in [
            ("session_id_1", "session_id_2"),
            ("session_id_3", "session_id_4"),
        ]:
            # make sure information does not leak between sessions
            self._test_composition_same_session_ids(
                prompt_1,
                prompt_2,
                num_generated_tokens,
                pipeline,
                session_id_1=session_id_pair[0],
                session_id_2=session_id_pair[1],
            )

    def _test_composition_same_session_ids(
        self,
        prompt_1,
        prompt_2,
        num_generated_tokens,
        pipeline,
        session_id_1,
        session_id_2,
    ):

        tokenizer = pipeline.tokenizer

        # make sure that running two prompts one after another
        # is identical to running a composition of those prompts
        out_1_ = self.run_pipeline(
            pipeline,
            sequences=prompt_1,
            session_ids=session_id_1,
            max_tokens=num_generated_tokens,
        )

        prompt_1_ = out_1_.sequences[0]

        out_1 = self.run_pipeline(
            pipeline,
            sequences=prompt_2,
            session_ids=session_id_1,
            max_tokens=num_generated_tokens,
        )

        prompt_composition = tokenizer.decode(
            tokenizer(prompt_1).input_ids
            + tokenizer(prompt_1_).input_ids
            + tokenizer(prompt_2).input_ids,
            skip_special_tokens=True,
        )
        out_2 = self.run_pipeline(
            pipeline,
            sequences=prompt_composition,
            session_ids=session_id_2,
            max_tokens=num_generated_tokens,
        )

        kv_cache_1 = pipeline.engine.kv_cache_storage.get(session_id_1)
        cache_array_1 = kv_cache_1.cached_inputs["past_key_values.0.key"][
            :, :, -kv_cache_1.total_num_processed_tokens :
        ]
        kv_cache_2 = pipeline.engine.kv_cache_storage.get(session_id_2)
        cache_array_2 = kv_cache_2.cached_inputs["past_key_values.0.key"][
            :, :, -kv_cache_2.total_num_processed_tokens :
        ]
        assert numpy.allclose(cache_array_1, cache_array_2, atol=PRECISION)
        assert out_1.sequences[0] == out_2.sequences[0]

    def _test_output(
        self,
        output: "TextGenerationOutput",  # noqa F821
        cache_session: DecoderKVCache,
        torch_ground_truth: Tuple[numpy.ndarray, ...],
        logits_threshold: Optional[float] = None,
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

        if logits_threshold:
            # if comparing the output from the model where
            # the kv cache has been filled, we expect the
            # maximum absolute difference between the logits
            # to be less than the threshold
            # (the threshold is established by running the
            # ONNX model in ONNXRuntime)
            assert abs(output.logits - target_logits).max() < logits_threshold
        else:
            # otherwise, we expect the logits to be exactly the same
            # as the target logits; the generated sequence should
            # also be the same as the target sequence, and finally
            # (if applicable) the kv cache should be the same as the
            # target kv cache
            assert numpy.allclose(output.logits, target_logits, atol=PRECISION)
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
            assert numpy.allclose(
                x[:, :, -start_index:-end_index, :], y, atol=PRECISION
            )

    @staticmethod
    def _pop_cache_session(pipeline: Pipeline) -> "DecoderKVCache":  # noqa F821
        # pop the last cache session from the kv cache storage
        # (this makes sure that we are only keeping the most
        # recent cache session in the storage, useful for testing)
        memory = pipeline.engine.kv_cache_storage._memory
        assert len(memory) == 1
        # access the only element in the memory dict
        session = list(memory.values())[0]
        # remove the session from the memory dict to always have
        pipeline.engine.kv_cache_storage.pop(session.id)
        return session
