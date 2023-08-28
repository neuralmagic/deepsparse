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


import numpy as np
import onnx
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.utils.helpers import (
    create_causal_mask,
    overwrite_onnx_model_inputs_for_kv_cache_models,
)
from deepsparse.utils.onnx import CACHE_INPUT_PREFIX
from sparsezoo import Model


def _initialize_kv_cache_state(model, length=0):
    # get one of the cache inputs
    cache_input = next(
        input
        for input in model.graph.input
        if input.name.startswith(CACHE_INPUT_PREFIX)
    )
    # read the shape of the cache input
    batch_size = cache_input.type.tensor_type.shape.dim[0].dim_value
    num_attention_heads = cache_input.type.tensor_type.shape.dim[1].dim_value
    hidden_dims = cache_input.type.tensor_type.shape.dim[3].dim_value

    # create a kv cache dictionary
    kv_cache = {
        input_.name: np.zeros(
            (batch_size, num_attention_heads, length, hidden_dims), dtype=np.float32
        )
        for input_ in model.graph.input
        if input_.name.startswith(CACHE_INPUT_PREFIX)
    }

    return kv_cache


START = 0  # global variable for dummy_callback


@pytest.mark.parametrize(
    "use_deepsparse_cache",
    [True, False],
)
@pytest.mark.parametrize(
    "model_stub, model_name, uses_bos_token",
    [
        (
            "zoo:nlg/text_generation/opt-1.3b/pytorch/"
            "huggingface/opt_pretrain/base-none",
            "facebook/opt-1.3b",
            True,
        ),
        (
            "zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
            "huggingface/bigpython_bigquery_thepile/base-none",
            "salesforce/codegen-350m-mono",
            False,
        ),
    ],
    scope="class",
)
# @pytest.mark.skip(
#     reason="Those tests are too heavy to " "run as a normal part of the CI."
# )
class TestTextGenerationPipeline:
    @pytest.fixture
    def setup(self, model_stub, model_name, uses_bos_token, use_deepsparse_cache):

        self.max_generated_tokens = 16
        self.model = Model(model_stub)
        self.use_deepsparse_cache = use_deepsparse_cache

        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_stub,
            sequence_length=64,
            prompt_processing_sequence_length=4,
            max_generated_tokens=self.max_generated_tokens,
            use_deepsparse_cache=self.use_deepsparse_cache,
        )
        short_prompt = "this"
        long_prompt = "this is a sample prompt that we will use to test the pipeline"

        # make sure that the short prompt will be only
        # processed by a single token engine
        # (DISABLED FOR NOW UNTIL WE HAVE ZOO CAUSAL MASK SUPPORT)
        # assert (
        #     len(pipeline.tokenizer.tokenize(short_prompt)) + int(uses_bos_token)
        #     < pipeline.prompt_processing_sequence_length
        # )
        # make sure that the long prompt will be processed by
        # single token and multiple token engines
        # (DISABLED FOR NOW UNTIL WE HAVE ZOO CAUSAL MASK SUPPORT)
        # assert (
        #     len(pipeline.tokenizer.tokenize(long_prompt)) + int(uses_bos_token)
        #     > pipeline.prompt_processing_sequence_length * 3
        # )

        yield pipeline, model_name, uses_bos_token, short_prompt, long_prompt

    def test_freeze_first_position(self, setup):
        # test whether we should be "freezing" the first token after
        # the kv cache is full
        pipeline, _, uses_bos_token, _, _ = setup
        assert pipeline.engine._freeze_first_position == uses_bos_token

    def test_model_output_sequences(self, setup):
        # test model output against sources of truth
        pipeline, model_name, _, short_prompt, long_prompt = setup

        output_sequences = pipeline(sequences=[short_prompt, long_prompt])

        # test against huggingface model
        output_hugging_face = self._get_output_huggingface(
            sequences=[short_prompt, long_prompt], model_name=model_name
        )
        assert short_prompt + output_sequences.sequences[0] == output_hugging_face[0]
        assert long_prompt + output_sequences.sequences[1] == output_hugging_face[1]

    def test_model_output_cache(self, setup):
        pipeline, model_name, _, short_prompt, long_prompt = setup
        if self.use_deepsparse_cache:
            pytest.skip(
                "Running pipeline with internal "
                "deepsparse cache will not result "
                "in meaningful cache entries."
            )
        self._test_cache_state(short_prompt, pipeline, model_name)
        self._test_cache_state(long_prompt, pipeline, model_name)

    def test_callback(self, setup):
        pipeline, *_ = setup

        def dummy_callback(token):
            global START
            START += 1
            return START < 3

        inputs = {
            "sequences": "def fib(a, b, accumulator=0)",
            "callback": dummy_callback,
            "return_logits": True,
        }

        outs = pipeline(**inputs)
        assert outs.logits.shape[1] == 3

    def test_model_session_storage(self, setup):
        pipeline, _, _, _, _ = setup

        # test whether the session storage is working correctly,
        # i.e storage memory is composable and there is no leakage
        # between sessions

        short_prompt = "this"
        long_prompt = "this is a sample prompt"

        output = pipeline(sequences=short_prompt, session_ids="session_one")
        intermediate_prompt_1 = output.sequences[0]
        out_1 = pipeline(sequences=long_prompt, session_ids="session_one")

        output = pipeline(sequences=long_prompt, session_ids="session_two")
        intermediate_prompt_2 = output.sequences[0]
        out_2 = pipeline(sequences=short_prompt, session_ids="session_two")

        out_3 = pipeline(
            sequences=short_prompt + intermediate_prompt_1 + long_prompt,
        )
        out_4 = pipeline(
            sequences=long_prompt + intermediate_prompt_2 + short_prompt,
        )

        assert out_1.sequences[0] == out_3.sequences[0]
        assert out_2.sequences[0] == out_4.sequences[0]

    def _test_cache_state(self, prompt, pipeline, model_name):
        # make sure that the cache state after running a prompt
        # is correct
        pipeline(sequences=prompt, session_ids="cache_state_test")
        cache_state_dict = pipeline.engine.kv_cache_storage.get(
            "cache_state_test"
        ).cached_inputs
        cache_state_list = [cache_state_dict[key] for key in cache_state_dict.keys()]

        # generate ground truth from ORT
        target_cache_state = self._get_cache_state_ort_kv_cache(
            model_onnx_path=self.model.deployment.get_file("model.onnx").path,
            sequence=prompt,
            model_name=model_name,
        )
        # get the number of processed prompt tokens
        num_prompt_tokens = len(pipeline.tokenizer.tokenize(prompt)) + int(
            pipeline.engine._freeze_first_position
        )

        for x, y in zip(cache_state_list, target_cache_state):
            """
            x will be a cache array
            [blank, blank, ..., prompt_cache_1, prompt_cache_2, ...,
             gen_token_cache_1, gen_token_cache_2, ...]
            we need to first remove blank entries and then keep the
            remaining prompt_cache entries (remove gen_token_cache entries)
            """
            first_non_blank_cache_entry = min(
                i for i in range(x.shape[2]) if np.count_nonzero(x[:, :, i, :])
            )
            x = x[:, :, first_non_blank_cache_entry:, :]
            x = x[:, :, :num_prompt_tokens, :]

            """
            y will be a cache array
            [blank, blank, ..., prompt_cache_1, prompt_cache_2, ...]
            we need to keep the prompt_cache entries only
            """
            y = y[:, :, -num_prompt_tokens:, :]

            assert np.allclose(x, y, atol=1e-4)

    def _get_output_huggingface(self, sequences, model_name):
        hf_outputs = []
        # setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # setup model
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # generate ground truth output
        for prompt in sequences:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            generated_ids = model.generate(
                input_ids, max_new_tokens=self.max_generated_tokens + 1
            )
            hf_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            hf_outputs.append(hf_output)
        return hf_outputs

    @staticmethod
    def _get_cache_state_ort_kv_cache(model_onnx_path, sequence, model_name):
        # setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # setup model and session
        # (run full sequence inference)
        overwrite_onnx_model_inputs_for_kv_cache_models(
            model_onnx_path, sequence_length=128, input_ids_length=128
        )
        sess = onnxruntime.InferenceSession(model_onnx_path)

        # get model inputs
        onnx_model = onnx.load(model_onnx_path, load_external_data=False)
        model_inputs = [x.name for x in onnx_model.graph.input]
        kv_cache = _initialize_kv_cache_state(model=onnx_model)

        inputs = tokenizer(
            sequence, return_tensors="np", padding="max_length", max_length=128
        )
        onnxruntime_inputs = dict(
            attention_mask=inputs["attention_mask"],
            input_ids=inputs["input_ids"],
            **kv_cache,
        )

        if "positions" in model_inputs:
            attention_mask = inputs["attention_mask"]
            positions = attention_mask.cumsum(1) * attention_mask - 1
            onnxruntime_inputs["positions"] = positions

        if "causal_mask" in model_inputs:
            causal_mask = create_causal_mask(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            onnxruntime_inputs["causal_mask"] = causal_mask

        # run inference and return the cache state
        outputs = sess.run(None, onnxruntime_inputs)
        logits, *kv_cache = outputs

        return kv_cache
