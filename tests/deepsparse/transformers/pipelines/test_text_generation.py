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

from transformers import AutoModelForCausalLM, AutoTokenizer

import pytest
from deepsparse import Pipeline


@pytest.mark.parametrize(
    "stub, model_name",
    [
        ("zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
        "huggingface/bigpython_bigquery_thepile/base-none",
         "Salesforce/Codegen-350M-mono"),
    ],
    scope="class",
)
class TestTextGenerationPipeline:
    @pytest.fixture
    def setup(self, stub, model_name):
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=stub,
            max_generated_tokens=16,
            prompt_processing_sequence_length=1, # TODO: Change to != 1
            use_deepsparse_cache=False,
        )
        short_prompt = "this is a sample prompt that"
        long_prompt = "this is a sample prompt that we will use to test the pipeline"

        # TODO: Enable it later
        # assert (
        #     len(pipeline.tokenizer.tokenize(short_prompt))
        #     < pipeline.prompt_processing_sequence_length
        # )
        # assert (
        #     len(pipeline.tokenizer.tokenize(long_prompt))
        #     > pipeline.prompt_processing_sequence_length * 3
        # )

        yield pipeline, model_name, short_prompt, long_prompt

    def test_model_output(self, setup):
        pipeline, model_name, short_prompt, long_prompt = setup

        output = pipeline(sequences=[short_prompt, long_prompt])
        output_1 = short_prompt + output.sequences[0]
        output_2 = long_prompt + output.sequences[1]

        self._test_against_huggingface(
            inputs=[short_prompt, long_prompt],
            outputs=[output_1, output_2],
            model_name=model_name,
        )
        self._test_against_no_kv_cache_model(
            inputs=[short_prompt, long_prompt],
            outputs=[output_1, output_2],
            model_name=model_name,
        )

    @staticmethod
    def _test_against_huggingface(inputs, outputs, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        for input, output in zip(inputs, outputs):
            input_ids = tokenizer(input, return_tensors="pt").input_ids
            generated_ids = model.generate(input_ids, max_new_tokens=16)
            hf_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(output, hf_output)

    @staticmethod
    def _test_against_no_kv_cache_model(inputs, outputs, model_name):
        pass
