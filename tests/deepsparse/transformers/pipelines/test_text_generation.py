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

from sparsezoo import Model
import pytest
import onnxruntime
import numpy
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod



class GroundTruthSource(ABC):
    def __init__(self, num_tokens_to_generate: int, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.num_tokens_to_generate = num_tokens_to_generate
        self.tokenizer = tokenizer

    # make this an abstract method
    @abstractmethod
    def tokenize(self, prompt: str) -> Dict[str, numpy.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, prompt: str) -> numpy.ndarray:
        raise NotImplementedError()


class ORTGroundTruthSource(GroundTruthSource):
    def __init__(self, model_stub: str, num_tokens_to_generate: int, model_name: str):
        super().__init__(num_tokens_to_generate, model_name)

        self.model_onnx_path = Model(model_stub).training.get_file("model.onnx").path
        self.session = onnxruntime.InferenceSession(self.model_onnx_path)

    def tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="np", padding="max_length", max_length=self.sequence_length)

    def __call__(self, prompt: str) -> numpy.ndarray:
        inputs = self.tokenize(prompt)
        logits = self.session.run(None, inputs)
        return logits


class TorchGroundTruthSource(GroundTruthSource):
    def __init__(self, num_tokens_to_generate: int, model_name: str):
        super().__init__(num_tokens_to_generate, model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize(self, prompt:str):
        return self.tokenizer(prompt, return_tensors="pt")

    def __call__(self, prompt: str) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]:
        # afaik it is not possible to get 'past_key_values' from the generate method, so we have to
        # run the model twice
        out = self.model.generate(self.tokenize(prompt).input_ids, max_new_tokens=self.num_tokens_to_generate, output_scores = True, return_dict_in_generate=True, use_cache=True)
        generated_logits = numpy.concatenate([[score.numpy() for score in out.scores]]) # (1, num_tokens_to_generate, vocab_size)

        out = self.model(**self.tokenize(prompt))
        prompt_logits = out.logits.detach().numpy() # (1, prompt_length, vocab_size)
        prompt_cache = [element.detach.numpy() for tupl in out.past_key_values for element in tupl] # List[(1, num_heads, past_length, head_dim)]

        return generated_logits, prompt_logits, prompt_cache


@pytest.mark.parametrize(
    "model_stub, model_name, uses_bos_token",
    [
        (
            "zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
            "huggingface/bigpython_bigquery_thepile/base-none",
            "salesforce/codegen-350m-mono",
            False,
        ),
    ],
    scope="class",
)
def test_ground_truth_sources(model_stub, model_name, uses_bos_token):
    num_tokens_generate = 256
    prompt = "This is a test prompt, that is used to generate some text"
    #ort_target_logits = ORTGroundTruthSource(model_stub=model_stub, num_tokens_to_generate=num_tokens_generate,model_name=model_name)(prompt)
    torch_target_generated_logits, torch_target_prompt_logits, torch_target_prompt_cache = TorchGroundTruthSource(num_tokens_to_generate = num_tokens_generate, model_name = model_name)(prompt)

    print(torch_target_logits)



