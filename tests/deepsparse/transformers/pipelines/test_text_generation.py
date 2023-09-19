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

from typing import List, Optional, Tuple, Union

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from tests.deepsparse.transformers.pipelines.helpers import (
    TorchGroundTruthSource,
    helper_test,
    parse_params,
)


# A sample config file requires the following arguments:
#     model_path: The path to the model to be tested
#                 (sparsezoo stub/local_path/remote_url)
#     model_name: The name of the hugging face model
#                 (to generate ground truth info)
#     num_tokens_generate: The number of tokens to generate
#     prompt: The prompt to use for testing
#     has_bos_token: Whether the model has a bos token
#     logits_threshold: The treshold for the max difference between the
#                        actual and the expected logits in the situations
#                        where they will not be able to match the ground
#                        truth (e.g. when the DeepSparse pipeline is
#                        running after the KV cache has been filled up).
#                        This value is established
#                        empirically for each combination of
#                        prompt/pipeline/num_generated tokens.
#     precision: The precision for the logits/kv_cache entries comparison
#     cache_management_type: The type of cache management to be tested.
#                            The available options are: "internal" and "external".
#     run_helper_tests: Whether to run the helper test for the pipeline. Helper tests
#                      check functionalities of the pipeline that are not directly
#                      on the hot path.
#     cadence: The cadence of the tests. The available options are:
#               "nightly" and "commit". By default, only the tests that have cadence
#               "commit" will be run in GHA.


# the user can specify the config file to be used for the tests
AVAILABLE_CONFIGS = [
    "tests/deepsparse/transformers/pipelines/configs/text_generation_gpt_neo.yaml",
    "tests/deepsparse/transformers/pipelines/configs/text_generation_opt.yaml",
    "tests/deepsparse/transformers/pipelines/configs/text_generation_opt_ort.yaml",
]


@pytest.fixture(scope="class")
def config(request):
    return request.param


@pytest.mark.parametrize("config", AVAILABLE_CONFIGS, indirect=["config"])
@pytest.mark.parametrize(
    "internal_kv_cache",
    [True, False],
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
        config,
        internal_kv_cache,
    ):
        params_dict, skip_reason = parse_params(config)
        if params_dict is None:
            pytest.skip(skip_reason)
        # set the params_dict as attributes of the test class
        for k, v in params_dict.items():
            setattr(self, k, v)

        if internal_kv_cache and "internal" not in self.cache_management_type:
            pytest.skip(
                "The tests for running the pipeline with "
                "internal kv cache management are disabled."
                f"Edit the config file: {self.pytestmark[1].args[1]}"
                "to enable them."
            )
        if not internal_kv_cache and "external" not in self.cache_management_type:
            pytest.skip(
                "The tests for running the pipeline with "
                "external kv cache management are disabled."
                f"Edit the config file: {self.pytestmark[1].args[1]}"
                "to enable them."
            )

        self.internal_kv_cache = internal_kv_cache

        # create torch ground source
        torch_source = TorchGroundTruthSource(
            num_tokens_to_generate=self.num_tokens_generate, model_name=self.model_name
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

        self.default_pipeline_kwargs = dict(
            task="text_generation",
            model_path=self.model_path,
            internal_kv_cache=self.internal_kv_cache,
            prompt_sequence_length=self.prompt_sequence_length,
            sequence_length=self.sequence_length,
        )
        self.default_pipeline = None

        assert self.prompt_sequence_length < prompt_length, (
            "The prompt processing sequence length "
            "must be smaller than the prompt length"
        )

    def get_pipeline(self, **kwargs):
        # if no kwargs provided, returns the cached "default"
        # pipeline that is used for most of the tests.
        # Otherwise, returns a pipeline with the given kwargs
        # (the default pipeline kwargs are updated with the
        # user-provided kwargs)
        if not kwargs:
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
        cache_session = pipeline.engine.kv_cache
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
        cache_session = pipeline.engine.kv_cache
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
        cache_session = pipeline.engine.kv_cache
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
        cache_session = pipeline.engine.kv_cache
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
        cache_session = pipeline.engine.kv_cache
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
        cache_session = pipeline.engine.kv_cache
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

    @helper_test
    def test_run_same_prompt_multiple_times(self, setup):
        # Test the scenario, where the same prompt is run multiple times
        # Every run should produce the same output
        pipeline = self.get_pipeline()

        output_1 = self.run_pipeline(pipeline)
        output_2 = self.run_pipeline(pipeline)

        assert output_1.sequences[0] == output_2.sequences[0]
        assert numpy.allclose(output_1.logits, output_2.logits, atol=self.precision)

    @helper_test
    def test_run_multiple_prompts_in_parallel(self, setup):
        # Test the scenario, where multiple prompts are run in parallel
        # Same two prompts should produce the same output
        pipeline = self.get_pipeline()

        output = self.run_pipeline(pipeline, sequences=[self.prompt, self.prompt])

        assert numpy.allclose(output.logits[0], output.logits[1], atol=self.precision)
        assert output.sequences[0] == output.sequences[1]

    @helper_test
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

    @helper_test
    def test_freeze_first_position(self, setup):
        # Test whether we should be "freezing" the first token after
        # the kv cache is full
        pipeline = self.get_pipeline()
        assert pipeline.engine.freeze_first_position == self.has_bos_token

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

            if target_logits.shape[1] < output.logits.shape[1]:
                output.logits = output.logits[:, : target_logits.shape[1]]
            assert abs(output.logits - target_logits).max() < logits_threshold
        else:
            # otherwise, we expect the logits to be exactly the same
            # as the target logits; the generated sequence should
            # also be the same as the target sequence, and finally
            # (if applicable) the kv cache should be the same as the
            # target kv cache
            assert numpy.allclose(output.logits, target_logits, atol=self.precision)
            assert self.prompt + output.sequences[0] == generated_text

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
            # x is (in general) composed of three arrays:
            # - padding cache entries (from 0 to -start_index)
            # - prompt cache entries (from -start_index to -end_index)
            # - generated cache entries (from -end_index to -1)
            # as target_cache only pertains to prompt cache entries, we need to
            # compare only the prompt cache entries in x with y
            assert numpy.allclose(
                x[:, :, -start_index:-end_index, :], y, atol=self.precision
            )
