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
for various scenerios.

A sample config file is a yaml that requires the following fields:
    cadence: The cadence of the tests. The available options are:
              "nightly", "weekly" and "commit". By default, only
              the tests that have cadence "commit" will be run
              in GHA. This parameter can be both a string or a
              list of strings.
    model_path: The path to the model to be tested
                (sparsezoo stub/hf model path/local_path)
    torch_model_name: The name of the torch model
                (to generate ground truth info)
    task: The task to be tested
                   (e.g. text-generation)
    prompt: The prompt to use for testing
    has_bos_token: Whether the model has a bos token
    precision: The precision for the logits/kv_cache entries
        comparison
    internal_kv_cache: The type of the internal KV cache
        management. Is a list that can contain the following
        values: [True], [False] or [True, False] (to test both
        external and internal KV cache management)
"""
import os
from typing import List, Tuple

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.pipelines.text_generation import TextGenerationOutput
from sparsezoo import Model
from tests.deepsparse.transformers.pipelines.integration_tests.helpers import (
    TorchGroundTruthSource,
    parse_params,
    validate_internal_kv_cache,
    validate_task,
)


CONFIGS_DIRECTORY = "tests/deepsparse/transformers/pipelines/integration_tests/configs"


@pytest.fixture()
def max_new_tokens() -> int:
    return 64


@pytest.mark.parametrize("params_dict", parse_params(CONFIGS_DIRECTORY))
@pytest.mark.parametrize(
    "internal_kv_cache",
    [True, False],
)
@pytest.mark.parametrize(
    "task",
    ["text-generation", "chat"],
)
class TestsIntegrationLLMsPipelines:
    """
    This test suite is meant to test the main scenarios of
    the text generation pipeline.
    """

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

    @pytest.fixture
    def setup(self, params_dict, max_new_tokens, internal_kv_cache, task):
        # set the params_dict as the class attributes
        for key, value in params_dict.items():
            setattr(self, key, value)
        # check whether the specified cache management type
        # is supported for testing (skip if not supported)
        self.internal_kv_cache: bool = validate_internal_kv_cache(
            internal_kv_cache, self.internal_kv_cache
        )
        self.task: str = validate_task(task, self.task)
        # create torch ground source
        torch_source = TorchGroundTruthSource(
            num_tokens_to_generate=max_new_tokens + 1,
            model_name=self.torch_model_name,
        )
        # create torch ground truth
        self.torch_ground_truth = torch_source(self.prompt)

        # specify the default pipeline kwargs
        self.default_pipeline_kwargs = dict(
            task=self.task,
            model_path=self.model_path,
            internal_kv_cache=self.internal_kv_cache,
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
        pipeline._debug = True
        output = pipeline(
            self.prompt,
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
            include_prompt_logits=True,
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
        output = pipeline(
            self.prompt,
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
            include_prompt_logits=True,
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
        pipeline._debug = True
        output = pipeline(
            self.prompt,
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
            include_prompt_logits=True,
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
        output = pipeline(
            self.prompt,
            max_new_tokens=self.max_new_tokens,
            output_scores=True,
            include_prompt_logits=True,
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
        model_path_no_cache = self._get_model_path_no_cache()
        pipeline = self.get_pipeline(
            model_path=model_path_no_cache, engine_type=engine_type
        )
        assert not pipeline.cache_support_enabled, (
            "This pipeline test inference using non-kv cache "
            "model and thus should not support kv cache"
        )

        output = pipeline(
            self.prompt, max_length=1, output_scores=True, include_prompt_logits=True
        )
        prompt_length = self.torch_ground_truth[1].shape[1]
        # prompt logits + one logit for the new generated token
        logits = output.generations[0].score[-(prompt_length + 1) :, :]
        # compute ground truth logits analogously
        generated_logits, prompt_logits, *_ = self.torch_ground_truth
        logits_gt = numpy.concatenate(
            [prompt_logits[0], generated_logits[0, :1, :]], axis=0
        )
        assert numpy.allclose(logits, logits_gt, atol=self.precision)

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

    def _get_model_path_no_cache(self):
        if not self.model_path.startswith("zoo:"):
            pytest.skip("For this test, for now only the zoo model is supported")
        model = Model(self.model_path)
        # fetch the necessary file names for pipeline creation
        required_file_names = [
            os.path.basename(file.name) for file in model.deployment.files
        ]
        training_directory = model.training
        onnx_model_name_no_cache = [
            os.path.basename(file.name)
            for file in model.training.files
            if file.name.endswith(".onnx")
        ][0]

        # check if 'training' exists,
        # if not, download the files
        if "training" not in os.listdir(model._path):
            for filename in required_file_names:
                # download the files to a training directory
                if filename.endswith(".data"):
                    # data files are typically stored in a deployment directory
                    # download them to training
                    file = model.deployment.get_file(filename)
                    assert (
                        file is not None
                    ), f"Unable to find file {filename} in model {model}"
                    file.name = file.name.replace("deployment", "training")
                    file.download()
                    continue

                if filename.endswith(".onnx"):
                    # instead of `model.onnx` the onnx_model_name_no_cache
                    # should be downloaded
                    filename = filename.replace("model.onnx", onnx_model_name_no_cache)

                file = training_directory.get_file(filename)
                assert (
                    file is not None
                ), f"Unable to find file {filename} in model {model}"
                file.download()
            # rename the model file to `model.onnx`
            os.rename(
                os.path.join(training_directory.path, onnx_model_name_no_cache),
                os.path.join(training_directory.path, "model.onnx"),
            )
        return training_directory._path
