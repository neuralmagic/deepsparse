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
from tests.deepsparse.transformers.pipelines.helpers import TorchGroundTruthSource


_PRECISION = 1e-3

NATURAL_LANGUAGE_PROMPT = """
Didn't know what time it was, the lights were low
I leaned back on my radio
Some cat was layin' down some rock 'n' roll
"Lotta soul," he said
Then the loud sound did seem to fade
Came back like a slow voice on a wave of phase
That weren't no DJ, that was hazy cosmic jive
"""

CODE_LANGUAGE_PROMPT = """
def Fibonacci(n):
    # Check if input is 0 then it will
    # print incorrect input
    if n < 0:
        print("Incorrect input")
    # Check if n is 0
    # then it will return 0
    elif n == 0:
        return 0
"""


@pytest.mark.parametrize(
    "internal_kv_cache",
    [True, False],
)
@pytest.mark.parametrize(
    "model_stub, "
    "model_name, "
    "uses_bos_token, "
    "prompt, "
    "logits_max_diff_kv_cache_has_been_filled",
    [
        (
            "zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
            "huggingface/bigpython_bigquery_thepile/base-none",
            "salesforce/codegen-350m-mono",
            False,
            CODE_LANGUAGE_PROMPT,
            13,
        ),
        (
            "zoo:nlg/text_generation/opt-1.3b/pytorch/huggingface/"
            "opt_pretrain/base-none",
            "facebook/opt-1.3b",
            True,
            NATURAL_LANGUAGE_PROMPT,
            3.9,
        ),
    ],
    scope="class",
)
@pytest.mark.skip(reason="Those tests are too heavy to run as a normal part of the CI.")
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
        uses_bos_token,
        prompt,
        logits_max_diff_kv_cache_has_been_filled,
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
        self.logits_max_diff_kv_cache_has_been_filled = (
            logits_max_diff_kv_cache_has_been_filled
        )
        self.internal_kv_cache = internal_kv_cache

        self.default_pipeline = None

        assert self.prompt_sequence_length < prompt_length, (
            "The prompt processing sequence length "
            "must be smaller than the prompt length"
        )

        yield model_name, uses_bos_token, torch_ground_truth

    def test_freeze_first_position(self, setup):
        # Test whether we should be "freezing" the first token after
        # the kv cache is full
        _, uses_bos_token, _ = setup
        pipeline = self.get_pipeline()
        assert pipeline.engine._freeze_first_position == uses_bos_token

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
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        cache_session = self._get_last_cache_session(pipeline)
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
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        cache_session = self._get_last_cache_session(pipeline)
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
            force_max_tokens=True,
            engine_type="onnxruntime",
        )
        output = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        cache_session = self._get_last_cache_session(pipeline)
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

        _, _, torch_ground_truth = setup
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            sequence_length=self.sequence_length,
            prompt_sequence_length=1,
            force_max_tokens=True,
            internal_kv_cache=self.internal_kv_cache,
        )
        output = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        cache_session = self._get_last_cache_session(pipeline)
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
            force_max_tokens=True,
            internal_kv_cache=self.internal_kv_cache,
        )
        output = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        cache_session = self._get_last_cache_session(pipeline)
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
            force_max_tokens=True,
            internal_kv_cache=self.internal_kv_cache,
        )
        output = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        cache_session = self._get_last_cache_session(pipeline)
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
            max_logits_difference_threshold=self.logits_max_diff_kv_cache_has_been_filled,  # noqa E501
        )

    def test_run_same_prompt_multiple_times(self, setup):
        # Test the scenario, where the same prompt is run multiple times
        # Every run should produce the same output
        pipeline = self.get_pipeline()

        output_1 = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        output_2 = pipeline(
            sequences=self.prompt,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )
        assert output_1.sequences[0] == output_2.sequences[0]
        assert numpy.allclose(output_1.logits, output_2.logits, atol=_PRECISION)

    def test_run_multiple_prompts_in_parallel(self, setup):
        # Test the scenario, where multiple prompts are run in parallel
        # Same two prompts should produce the same output
        pipeline = self.get_pipeline()

        output = pipeline(
            sequences=[self.prompt, self.prompt],
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=self.num_tokens_generate,
        )

        assert numpy.allclose(output.logits[0], output.logits[1], atol=_PRECISION)
        assert output.sequences[0] == output.sequences[1]

    def test_num_generated_predictions(self, setup):
        # Test the scenario, where multiple predictions are generated
        # from the same prompt
        pipeline = self.get_pipeline()

        output_sequences = pipeline(
            sequences=[self.prompt], num_generated_predictions=2
        )
        assert len(output_sequences.sequences[0]) == 2

        output_sequences = pipeline(
            sequences=[self.prompt, self.prompt], num_generated_predictions=2
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
        prompt_2 = "still this prompt should not"
        num_generated_tokens = 64
        uses_bos_token = setup[1]

        self._test_run_with_same_session_ids(
            prompt_1,
            prompt_2,
            num_generated_tokens,
            uses_bos_token,
            multi_token_prefill=False,
        )
        self._test_run_with_same_session_ids(
            prompt_1,
            prompt_2,
            num_generated_tokens,
            uses_bos_token,
            multi_token_prefill=True,
        )

        pipeline = Pipeline.create(
            task="text_generation",
            use_deepsparse_cache=self.use_deepsparse_cache,
        )
        assert hasattr(
            pipeline, "session"
        ), "Pipeline does not have session context manager"

    def _test_run_with_same_session_ids(
        self,
        prompt_1,
        prompt_2,
        num_generated_tokens,
        uses_bos_token,
        multi_token_prefill,
    ):
        pipeline = self.get_pipeline(
            task="text_generation",
            model_path=self.model_stub,
            prompt_sequence_length=self.prompt_sequence_length
            if multi_token_prefill
            else 1,
            force_max_tokens=True,
            internal_kv_cache=self.internal_kv_cache,
        )

        # make sure information does not leak between sessions
        self._test_composition_same_session_ids(
            prompt_1,
            prompt_2,
            num_generated_tokens,
            pipeline,
            uses_bos_token,
            session_id_1="test_1",
            session_id_2="test_2",
        )
        self._test_composition_same_session_ids(
            prompt_1,
            prompt_2,
            num_generated_tokens,
            pipeline,
            uses_bos_token,
            session_id_1="test_3",
            session_id_2="test_4",
        )

    @staticmethod
    def _test_composition_same_session_ids(
        prompt_1,
        prompt_2,
        num_generated_tokens,
        pipeline,
        uses_bos_token,
        session_id_1,
        session_id_2,
    ):

        tokenizer = pipeline.tokenizer

        # make sure that running two prompts one after another
        # is identical to running a composition of those prompts
        out_1_ = pipeline(
            sequences=prompt_1,
            session_ids=session_id_1,
            max_tokens=num_generated_tokens,
            return_logits=True,
            include_prompt_logits=True,
        )
        prompt_1_ = out_1_.sequences[0]
        out_1 = pipeline(
            sequences=prompt_2,
            session_ids=session_id_1,
            return_logits=True,
            max_tokens=num_generated_tokens,
            include_prompt_logits=True,
        )
        if uses_bos_token:
            prompt_composition = tokenizer.decode(
                tokenizer(prompt_1).input_ids
                + tokenizer(prompt_1_).input_ids[1:]
                + tokenizer(prompt_2).input_ids[1:],
                skip_special_tokens=True,
            )
        else:
            prompt_composition = tokenizer.decode(
                tokenizer(prompt_1).input_ids
                + tokenizer(prompt_1_).input_ids
                + tokenizer(prompt_2).input_ids,
                skip_special_tokens=True,
            )
        out_2 = pipeline(
            sequences=prompt_composition,
            session_ids=session_id_2,
            return_logits=True,
            include_prompt_logits=True,
            max_tokens=num_generated_tokens,
        )

        assert out_1.sequences[0] == out_2.sequences[0]

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
            assert numpy.allclose(output.logits, target_logits, atol=_PRECISION)
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
                x[:, :, -start_index:-end_index, :], y, atol=_PRECISION
            )

    @staticmethod
    def _get_last_cache_session(pipeline: Pipeline) -> "DecoderKVCache":  # noqa F821
        memory = pipeline.engine.kv_cache_storage._memory
        assert len(memory) == 1
        # access the only element in the memory dict
        session = list(memory.values())[0]
        # remove the session from the memory dict to always have
        pipeline.engine.kv_cache_storage.pop(session.id)
        return session
