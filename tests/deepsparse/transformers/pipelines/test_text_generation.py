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
    "model_path, model_name, engine_type",
    [
        ("/home/ubuntu/damian/sparseml/deployment_opt",
         "facebook/opt-350m",
         "onnxruntime"),
    ],
    scope="class",
)
class TestTextGenerationPipeline:
    @pytest.fixture
    def setup(self, model_path, model_name, engine_type):
        pipeline = Pipeline.create(
            task="text_generation",
            model_path=model_path,
            engine_type = engine_type,
            max_generated_tokens=16,
            prompt_processing_sequence_length=128,
            use_deepsparse_cache=False,
        )
        short_prompt = "this is a"
        long_prompt = "this is a sample prompt that we will use to test the pipeline"

        assert (
            len(pipeline.tokenizer.tokenize(short_prompt))
            < pipeline.prompt_processing_sequence_length
        )
        assert (
            len(pipeline.tokenizer.tokenize(long_prompt))
            > pipeline.prompt_processing_sequence_length * 3
        )

        yield pipeline, model_name, short_prompt, long_prompt

    def test_model_output(self, setup):
        pipeline, model_name, short_prompt, long_prompt = setup

        # Test against hugingface model
        output_model_cache = pipeline(sequences=[short_prompt, long_prompt])
        output_hugging_face = self._get_output_huggingface(sequences= [short_prompt, long_prompt], model_name=model_name)
        assert short_prompt + output_model_cache.sequences[0] == output_hugging_face[0]
        assert long_prompt + output_model_cache.sequences[1] == output_hugging_face[1]



        # self._test_against_no_kv_cache_model(
        #     inputs=[short_prompt, long_prompt],
        #     outputs=[output_1, output_2],
        #     model_name=model_name,
        # )

    @staticmethod
    def _get_output_huggingface(sequences, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
        outputs = []
        for input in sequences:
            input_ids = tokenizer(input, return_tensors="pt").input_ids
            generated_ids = model.generate(input_ids, max_new_tokens=16)
            hf_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            outputs.append(hf_output)
        return outputs


    @staticmethod
    def _test_against_no_kv_cache_model(inputs, outputs, model_name):
        pass

    def test_freeze(self, setup):
        pass