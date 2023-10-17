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
"""
A sample config file requires the following arguments:
    model_path: The path to the model to be tested
                (sparsezoo stub/hf model path/local_path)
    model_name: The name of the hugging face model
                (to generate ground truth info)
    pipeline_type: The type of the pipeline to be tested
                   (e.g. text-generation)
    num_tokens_generate: The number of tokens to generate
    prompt: The prompt to use for testing
    has_bos_token: Whether the model has a bos token
    logits_threshold: The treshold for the max difference between the
                       actual and the expected logits in the situations
                       where they will not be able to match the ground
                       truth (e.g. when the DeepSparse pipeline is
                       running after the KV cache has been filled up).
                       This value is established
                       empirically for each combination of
                       prompt/pipeline/num_generated tokens.
    precision: The precision for the logits/kv_cache entries comparison
    cache_management_type: The type of cache management to be tested.
                           The available options are: "internal" and "external".
    run_helper_tests: Whether to run the helper test for the pipeline. Helper tests
                     check functionalities of the pipeline that are not directly
                     on the hot path.
    cadence: The cadence of the tests. The available options are:
              "nightly" and "commit". By default, only the tests that have cadence
              "commit" will be run in GHA.
"""
import inspect
from typing import List, Optional, Tuple

import numpy
from transformers import GenerationConfig

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.pipelines.text_generation import TextGenerationOutput
from deepsparse.transformers.utils.helpers import prepends_bos_token
from tests.deepsparse.transformers.pipelines.helpers import (
    TorchGroundTruthSource,
    helper_test,
    parse_params,
    validate_cache_management_type,
)


# the user can specify the config file to be used for the tests
# TODO: add more configs
# TODO: add explanation
AVAILABLE_CONFIGS = [
    "tests/deepsparse/transformers/pipelines/configs/gpt_neo.yaml",
    # "tests/deepsparse/transformers/pipelines/configs/text_generation_opt.yaml",
    # "tests/deepsparse/transformers/pipelines/configs/text_generation_codegen.yaml",
]


@pytest.fixture
def config(request):
    return request.param


@pytest.mark.parametrize("config", AVAILABLE_CONFIGS, indirect=["config"])
@pytest.mark.parametrize(
    "internal_kv_cache",
    [True, False],
)
class TestTextGenerationPipeline:
    """
    This test suite is meant to test the main scenarios of
    the text generation pipeline.
    """

    @pytest.fixture
    def pipeline_type(self):
        return "text-generation"

    def get_pipeline(self, **kwargs) -> Pipeline:
        """
        If no kwargs provided, returns the cached "default"
        pipeline that is used for most of the tests.
        Otherwise, returns a pipeline with the given kwargs
        (the default pipeline kwargs are updated with the
        user-provided kwargs)

        :param kwargs: the optional kwargs to be used to
            create the pipeline (if not provided, the cached
            "default" pipeline is returned)
        :return: the appropriate pipeline
        """
        if not kwargs:
            if self.default_pipeline is None:
                self.default_pipeline = Pipeline.create(**self.default_pipeline_kwargs)
            return self.default_pipeline

        # return a pipeline with the updated default kwargs
        updated_kwargs = self.default_pipeline_kwargs.copy()
        updated_kwargs.update(kwargs)
        return Pipeline.create(**updated_kwargs)

    def run_pipeline(self, pipeline: Pipeline, **kwargs) -> TextGenerationOutput:
        """
        Run the pipeline and return the output

        :param pipeline: the pipeline to be run
        :param kwargs: the optional kwargs to be used to
            run the pipeline
        :return: the pipeline output
        """
        sequences = kwargs.get("sequences", self.prompt)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        do_sample = kwargs.get("do_sample", False)
        streaming = kwargs.get("streaming", False)

        config = GenerationConfig(
            output_scores=True,
            max_length=self.num_tokens_generate,
            top_k=0,
            top_p=0.0,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
        )
        return pipeline(
            sequences=sequences,
            force_max_tokens=True,
            include_prompt_logits=True,
            generation_config=config,
            streaming=streaming,
        )

    @pytest.fixture
    def setup(self, config, internal_kv_cache, pipeline_type):
        params_dict, skip_reason = parse_params(config)
        if params_dict is None:
            # skip the test if the config file is not available
            pytest.skip(skip_reason)
        # set the params_dict as the class attributes
        for key, value in params_dict.items():
            setattr(self, key, value)
        # check whether the internal kv cache is supported for testing
        # (skip if not supported)
        self.internal_kv_cache: bool = validate_cache_management_type(
            internal_kv_cache, self.cache_management_type
        )
        # check whether the pipeline_type is supported for testing
        # (skip if not supported)
        if pipeline_type not in self.pipeline_type:
            pytest.skip(
                f"Pipeline type: {self.pipeline_type} "
                f"does not match the current type: {pipeline_type}"
            )

        # create torch ground source
        torch_source = TorchGroundTruthSource(
            num_tokens_to_generate=self.num_tokens_generate, model_name=self.model_name
        )
        # create torch ground truth
        self.torch_ground_truth = torch_source(self.prompt)
        prompt_length = self.torch_ground_truth[1].shape[1]

        # sequence_length that assures that the KV cache will not be filled up
        self.sequence_length = 2 * prompt_length + self.num_tokens_generate
        # sequence_length that assures that the KV cache will be filled up
        self.sequence_length_short = self.num_tokens_generate

        # prompt_sequence_length used for the multi-token prefill scenario
        self.prompt_sequence_length = prompt_length // 4
        assert self.prompt_sequence_length < prompt_length, (
            "The prompt processing sequence length "
            "must be smaller than the prompt length"
        )

        self.default_pipeline_kwargs = dict(
            task=pipeline_type,
            model_path=self.model_path,
            internal_kv_cache=self.internal_kv_cache,
            prompt_sequence_length=self.prompt_sequence_length,
            sequence_length=self.sequence_length,
        )
        self.default_pipeline = None

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

        pipeline = self.get_pipeline(
            prompt_sequence_length=1,
            engine_type="onnxruntime",
        )
        pipeline._debug = True
        output = self.run_pipeline(pipeline)

        assert output.total_num_processed_tokens[0] < self.sequence_length, (
            "The total number of processed tokens must be smaller than the "
            "sequence length"
        )
        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
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
        pipeline = self.get_pipeline(
            engine_type="onnxruntime",
        )
        pipeline._debug = True
        output = self.run_pipeline(pipeline)

        assert output.total_num_processed_tokens[0] < self.sequence_length
        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
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

        pipeline = self.get_pipeline(
            sequence_length=self.sequence_length_short,
            engine_type="onnxruntime",
        )
        pipeline._debug = True
        output = self.run_pipeline(pipeline)

        assert output.total_num_processed_tokens[0] > self.sequence_length_short, (
            "for this scenario, the kv cache should be full: "
            "the total number of processed tokens should be "
            "greater than the sequence length"
        )

        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
            logits_threshold=self.logits_threshold,
        )

    def test_deepsparse_single_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by single-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed externally or internally

        pipeline = self.get_pipeline(
            prompt_sequence_length=1,
        )
        pipeline._debug = True
        output = self.run_pipeline(pipeline)

        assert output.total_num_processed_tokens[0] < self.sequence_length, (
            "The total number of processed tokens must be smaller than the "
            "sequence length"
        )
        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
            # disable kv cache validation if using internal kv cache
            run_kv_cache_validation=not self.internal_kv_cache,
        )

    def test_deepsparse_multi_token_prefill(self, setup):
        # Test the pipeline that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is never filled up
        # 3. KV Cache managed internally or externally

        pipeline = self.get_pipeline()
        pipeline._debug = True
        output = self.run_pipeline(pipeline)

        assert output.total_num_processed_tokens[0] < self.sequence_length

        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
            # disable kv cache validation if using internal kv cache
            run_kv_cache_validation=not self.internal_kv_cache,
        )

    def test_deepsparse_generation_after_kv_cache_has_been_filled(self, setup):
        # Test the deepsparse that uses deepsparse engine. The test covers the
        # following scenario:
        # 1. Prompt preprocessing is performed by multi-token engine
        # 2. The KV Cache is filled up (old entries are removed)
        # 3. KV Cache managed internally or externally

        pipeline = self.get_pipeline(
            sequence_length=self.sequence_length_short,
        )
        pipeline._debug = True
        output = self.run_pipeline(pipeline)

        assert output.total_num_processed_tokens[0] > self.sequence_length_short, (
            "for this scenario, the kv cache should be full: "
            "the total number of processed tokens should be "
            "greater than the sequence length"
        )

        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
            logits_threshold=self.logits_threshold,
            run_kv_cache_validation=not self.internal_kv_cache,
        )

    @helper_test
    def test_freeze_first_position(self, setup):
        # Test whether we should be "freezing" the first token after
        # the kv cache is full
        pipeline = self.get_pipeline()
        assert prepends_bos_token(pipeline.tokenizer) == self.has_bos_token

    @helper_test
    def test_run_same_prompt_multiple_times(self, setup):
        # Test the scenario, where the same prompt is run multiple times
        # Every run should produce the same output
        pipeline = self.get_pipeline()
        output_1 = self.run_pipeline(pipeline)
        output_2 = self.run_pipeline(pipeline)

        assert output_1.generations[0].text == output_2.generations[0].text
        assert numpy.allclose(
            output_1.generations[0].score,
            output_2.generations[0].score,
            atol=self.precision,
        )

    @helper_test
    def test_run_multiple_prompts_in_parallel(self, setup):
        # Test the scenario, where multiple prompts are run in parallel
        # Same two prompts should produce the same output
        pipeline = self.get_pipeline()
        output = self.run_pipeline(pipeline, sequences=[self.prompt, self.prompt])

        logits_0 = output.generations[0].score
        sequence_0 = output.generations[0].text

        logits_1 = output.generations[1].score
        sequence_1 = output.generations[1].text

        assert numpy.allclose(logits_0, logits_1, atol=self.precision)
        assert sequence_0 == sequence_1

    @helper_test
    def test_num_generated_predictions(self, setup):
        # Test the scenario, where multiple predictions are generated
        # from the same prompt
        pipeline = self.get_pipeline()

        output_sequences = self.run_pipeline(
            pipeline, sequences=[self.prompt], num_return_sequences=2
        )
        assert len(output_sequences.generations) == 1
        assert len(output_sequences.generations[0]) == 2

        output_sequences = self.run_pipeline(
            pipeline, sequences=[self.prompt, self.prompt], num_return_sequences=2
        )
        assert len(output_sequences.generations) == 2

        for generation in output_sequences.generations:
            assert len(generation) == 2

    @helper_test
    def test_token_generation_deterministic(self, setup):
        pipeline = self.get_pipeline()
        inference = self.run_pipeline(pipeline, num_return_sequences=3, do_sample=False)
        generations = inference.generations
        # Output should be the same from one another
        text_outputs = [x.text for x in generations[0]]
        assert len(set(text_outputs)) == 1

    @helper_test
    def test_token_generation_non_deterministic(self, setup):
        pipeline = self.get_pipeline()
        inference = self.run_pipeline(pipeline, num_return_sequences=3, do_sample=True)
        generations = inference.generations
        # Output should be different from one another
        text_outputs = [x.text for x in generations[0]]
        assert len(set(text_outputs)) == 3

    @helper_test
    def test_streaming_mode_returns_generator(self, setup):
        pipeline = self.get_pipeline(prompt_sequence_length=1)
        response_generator = self.run_pipeline(pipeline, streaming=True)

        assert inspect.isgenerator(
            response_generator
        ), "Pipeline should return a generator in streaming mode"

        assert all(
            isinstance(response, pipeline.output_schema)
            for response in response_generator
        ), "Pipeline should return a generator of output_schema \
               objects in streaming mode"

    def _test_output(
        self,
        output: TextGenerationOutput,
        torch_ground_truth: Tuple[numpy.ndarray, ...],
        logits_threshold: Optional[float] = None,
        run_kv_cache_validation: bool = True,
    ):

        (
            generated_logits,
            prompt_logits,
            prompt_kv_cache,
            generated_text,
        ) = torch_ground_truth

        # concatenate target prompt_logits and generated_logits
        target_logits = numpy.concatenate([prompt_logits, generated_logits], axis=1)
        # get the logits of the generated sequence
        score = output.generations[0].score

        if logits_threshold:
            # if comparing the output from the model where
            # the kv cache has been filled, we expect the
            # maximum absolute difference between the logits
            # to be less than the threshold
            # (the threshold is established by running the
            # ONNX model in ONNXRuntime)
            target_logits = target_logits[0]
            if target_logits.shape[0] < score.shape[0]:
                score = score[: target_logits.shape[0], :]
            assert abs(score - target_logits).max() < logits_threshold
        else:
            # otherwise, we expect the logits to be exactly the same
            # as the target logits; the generated sequence should
            # also be the same as the target sequence
            assert numpy.allclose(score, target_logits[0], atol=self.precision)
            assert self.prompt + output.generations[0].text == generated_text

            if hasattr(output, "kv_cache_state") and run_kv_cache_validation:
                # (if applicable) the kv cache should be the same as the
                # target kv cache
                expected_cache = list(output.kv_cache_state[0].values())
                total_num_processed_tokens = output.total_num_processed_tokens[0]
                self._test_kv_cache_state(
                    expected_cache=expected_cache,
                    target_cache=prompt_kv_cache,
                    total_num_processed_tokens=total_num_processed_tokens,
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
