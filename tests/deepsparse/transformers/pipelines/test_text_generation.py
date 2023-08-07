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

import numpy as np
import onnx
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer

import pytest
from deepsparse import Pipeline
from deepsparse.transformers.utils.helpers import (
    create_causal_mask,
    overwrite_onnx_model_inputs,
)


def _initialize_kv_cache_state(model, length=0):
    # get one of the cache inputs
    cache_input = next(
        input for input in model.graph.input if input.name.startswith("past_key_values")
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
        if input_.name.startswith("past_key_values")
    }

    return kv_cache


@pytest.mark.parametrize(
    "use_deepsparse_cache",
    [True, False],
)
@pytest.mark.parametrize(
    # TODO: Change to stubs
    "model_path, model_name, uses_bos_token",
    [
        ("/home/ubuntu/damian/sparseml/deployment_opt", "facebook/opt-350m", True),
        (
            "/home/ubuntu/damian/sparseml/deployment_codegen",
            "salesforce/codegen-350m-multi",
            False,
        ),
    ],
    scope="class",
)
class TestTextGenerationPipeline:
    @pytest.fixture
    def setup(self, model_path, model_name, uses_bos_token, use_deepsparse_cache):
        self.max_generated_tokens = 16
        self.use_deepsparse_cache = use_deepsparse_cache
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_path,
            sequence_length=32,
            prompt_processing_sequence_length=4,
            max_generated_tokens=self.max_generated_tokens,
            use_deepsparse_cache=use_deepsparse_cache,
        )
        short_prompt = "this"
        long_prompt = "this is a sample prompt that we will use to test the pipeline"

        # make sure that the short prompt will be only
        # processed by a single token engine
        assert (
            len(pipeline.tokenizer.tokenize(short_prompt)) + int(uses_bos_token)
            < pipeline.prompt_processing_sequence_length
        )
        # make sure that the long prompt will be processed by
        # single token and multiple token engines
        assert (
            len(pipeline.tokenizer.tokenize(long_prompt)) + int(uses_bos_token)
            > pipeline.prompt_processing_sequence_length * 3
        )

        yield pipeline, model_name, uses_bos_token, short_prompt, long_prompt

    def test_freeze(self, setup):
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
        if not self.use_deepsparse_cache:
            assert long_prompt + output_sequences.sequences[1] == output_hugging_face[1]

    def test_model_output_cache(self, setup):
        pipeline, model_name, _, short_prompt, long_prompt = setup
        if not self.use_deepsparse_cache:
            self._test_cache_state(short_prompt, pipeline, model_name)
            self._test_cache_state(long_prompt, pipeline, model_name)

    def _test_cache_state(self, prompt, pipeline, model_name):
        # make sure that the cache state after running a prompt
        # is correct

        pipeline(sequences=prompt)
        cache_state_dict = pipeline.engine.kv_cache.cached_inputs
        cache_state_list = [cache_state_dict[key] for key in cache_state_dict.keys()]

        # generate ground truth from ORT
        target_cache_state = self._get_cache_state_ort_kv_cache(
            model_onnx_path=os.path.join(pipeline._model_path, "model.onnx"),
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
                input_ids, max_new_tokens=self.max_generated_tokens
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
        overwrite_onnx_model_inputs(
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
