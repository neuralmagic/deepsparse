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
This test suite consumes config files to test the text generation pipeline
for various scenarios.

A sample config file is a yaml that requires the following fields:
    cadence: The cadence of the tests. The available options are:
              "nightly", "weekly" and "commit". By default, only
              the tests that have cadence "commit" will be run
              in GHA. This parameter can be both a string or a
              list of strings.
    model_path: The path to the model to be tested
                (sparsezoo stub/hf model path/local_path)
    model_name_no_kv_cache: The name of the onnx model without
                            the KV cache support
    torch_model_name: The name of the torch model
                (to generate ground truth info)
    prompt: The prompt to use for testing
    precision: The precision for the logits/kv_cache entries
        comparison
    internal_kv_cache: The type of the internal KV cache
        management. Is a list that can contain the following
        values: [True], [False] or [True, False] (to test both
        external and internal KV cache management)
"""
from typing import List, Tuple

import numpy

import pytest
from deepsparse.pipeline import Pipeline
from deepsparse.transformers.pipelines.text_generation import (
    TextGenerationPipeline,
    TextGenerationPipelineNoCache,
)
from deepsparse.transformers.schemas.text_generation_schemas import TextGenerationOutput
from tests.deepsparse.transformers.text_generation.integration_tests.helpers import (
    TorchGroundTruthSource,
    parse_params,
    validate_internal_kv_cache,
)


CONFIGS_DIRECTORY = (
    "tests/deepsparse/transformers/text_generation/integration_tests/configs"
)


@pytest.fixture()
def max_new_tokens() -> int:
    return 64


@pytest.mark.parametrize("params_dict", parse_params(CONFIGS_DIRECTORY))
@pytest.mark.parametrize(
    "internal_kv_cache",
    [True, False],
)
class TestsIntegrationLLMsPipelines:
    """
    This test suite is meant to test the main scenarios of
    the text generation pipeline.
    """

    def get_pipeline(self, kv_cache_support=True, **kwargs) -> Pipeline:
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
        # TODO: This if statement should disappear once
        # the TextGenerationPipeline contains the
        # non-kv-cache version of the pipeline
        text_generation_pipeline_class = (
            TextGenerationPipeline
            if kv_cache_support
            else TextGenerationPipelineNoCache
        )
        if not kwargs:
            if self.default_pipeline is None:
                self.default_pipeline = text_generation_pipeline_class(
                    **self.default_pipeline_kwargs
                )
            return self.default_pipeline

        # return a pipeline with the updated default kwargs
        updated_kwargs = self.default_pipeline_kwargs.copy()
        updated_kwargs.update(kwargs)
        return text_generation_pipeline_class(**updated_kwargs)

    @pytest.fixture
    def setup(self, params_dict, max_new_tokens, internal_kv_cache):
        # set the params_dict as the class attributes
        for key, value in params_dict.items():
            setattr(self, key, value)
        # check whether the specified cache management type
        # is supported for testing (skip if not supported)
        self.internal_kv_cache: bool = validate_internal_kv_cache(
            internal_kv_cache, self.internal_kv_cache
        )
        # create torch ground source
        torch_source = TorchGroundTruthSource(
            num_tokens_to_generate=max_new_tokens + 1,
            model_name=self.torch_model_name,
        )
        # create torch ground truth
        self.torch_ground_truth = torch_source(self.prompt)

        # specify the default pipeline kwargs
        self.default_pipeline_kwargs = dict(
            model_path=self.model_path,
            internal_kv_cache=self.internal_kv_cache,
            force_max_tokens=True,
        )
        self.default_pipeline = None
        self.max_new_tokens = max_new_tokens

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
        output = pipeline(
            prompt=self.prompt,
            include_prompt_logits=True,
            generation_kwargs=dict(
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
            ),
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
        pipeline = self.get_pipeline(engine_type="onnxruntime")
        output = pipeline(
            prompt=self.prompt,
            include_prompt_logits=True,
            generation_kwargs=dict(
                max_new_tokens=self.max_new_tokens, output_scores=True
            ),
        )

        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
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

        output = pipeline(
            prompt=self.prompt,
            include_prompt_logits=True,
            generation_kwargs=dict(
                max_new_tokens=self.max_new_tokens, output_scores=True
            ),
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
        output = pipeline(
            prompt=self.prompt,
            include_prompt_logits=True,
            generation_kwargs=dict(
                max_new_tokens=self.max_new_tokens, output_scores=True
            ),
        )

        self._test_output(
            output=output,
            torch_ground_truth=self.torch_ground_truth,
            # disable kv cache validation if using internal kv cache
            run_kv_cache_validation=not self.internal_kv_cache,
        )

    def test_inference_no_kv_cache_deepsparse(self, setup):
        self._test_inference_no_kv_cache(engine_type="deepsparse")

    def test_inference_no_kv_cache_ort(self, setup):
        self._test_inference_no_kv_cache(engine_type="onnxruntime")

    def _test_inference_no_kv_cache(self, engine_type):
        pipeline = self.get_pipeline(
            onnx_model_name=self.model_name_no_kv_cache,
            kv_cache_support=False,
            engine_type=engine_type,
        )

        output = pipeline(
            prompt=[self.prompt, self.prompt],
            include_prompt_logits=True,
            generation_kwargs=dict(output_scores=True),
        )

        # logits -> prompt logits + one logit for the new generated token
        generated_logits, prompt_logits, *_ = self.torch_ground_truth
        logits_gt = numpy.concatenate(
            [prompt_logits[0], generated_logits[0, :1, :]], axis=0
        )
        for gen in output.generations:
            assert numpy.allclose(gen.score, logits_gt, atol=self.precision)

    def _test_output(
        self,
        output: TextGenerationOutput,
        torch_ground_truth: Tuple[numpy.ndarray, ...],
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

        # we expect the logits to be exactly the same
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
