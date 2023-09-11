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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy
import onnx
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepsparse.transformers.utils.helpers import create_causal_mask
from deepsparse.utils.onnx import (
    CACHE_INPUT_PREFIX,
    overwrite_onnx_model_inputs_for_kv_cache_models,
)
from sparsezoo import Model


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


def uses_bos_token(model_name):
    return "opt" in model_name


def generate_pytest_params(stubs_to_test, cache_management_type, logits_thresholds):
    # process cache_management_type
    assert isinstance(
        cache_management_type, list
    ), "cache_management_type should be a list"
    assert len(cache_management_type) > 0, "cache_management_type should not be empty"
    assert all(
        [cache_type in ["internal", "external"] for cache_type in cache_management_type]
    ), (
        "cache_management_type should be a list "
        "of strings that are either 'internal' or 'external'"
    )
    cache_to_test = [cache_type == "internal" for cache_type in cache_management_type]

    # process stubs_to_test
    assert all([len(param) == 3 for param in stubs_to_test]), (
        "stubs_to_test should be a list of tuples in the form: "
        "[(model_stub, model_name, prompt), ...]"
    )

    # process logits_thresholds
    if logits_thresholds is None:
        logits_thresholds = [None] * len(stubs_to_test)
    else:
        assert len(logits_thresholds) == len(
            stubs_to_test
        ), "logits_thresholds should have the same length as stubs_to_test"

    for i, (param, threshold) in enumerate(zip(stubs_to_test, logits_thresholds)):
        model_name = param[1]
        stubs_to_test[i] = tuple(list(param) + [uses_bos_token(model_name), threshold])

    return cache_to_test, stubs_to_test


class GroundTruthSource(ABC):
    def __init__(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

    @abstractmethod
    def tokenize(self, prompt: str) -> Dict[str, Any]:
        """
        :param prompt: The prompt to tokenize
        :return: A dictionary of tokenized inputs
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, prompt: str) -> Any:
        """
        :param prompt: The prompt to generate from
        :return: Ground truth logits / cache state
        """
        raise NotImplementedError()


class ORTGroundTruthSource(GroundTruthSource):
    """
    An object that generates ground truth logits and
    cache states from a prompt. This object cannot
    generate tokens in an autoregressive manner, and thus
    will only output prompt logits and prompt cache state
    """

    def __init__(
        self,
        model_stub: str,
        model_name: str,
        sequence_length: int = 256,
    ):
        super().__init__(model_name)

        self.model_onnx_path = Model(model_stub).deployment.get_file("model.onnx").path
        overwrite_onnx_model_inputs_for_kv_cache_models(
            self.model_onnx_path,
            sequence_length=sequence_length,
            input_ids_length=sequence_length,
        )
        self.sequence_length = sequence_length
        self.session = onnxruntime.InferenceSession(self.model_onnx_path)
        self.model_inputs = [
            x.name
            for x in onnx.load(
                self.model_onnx_path, load_external_data=False
            ).graph.input
        ]

    def tokenize(self, prompt: str):
        return self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            max_length=self.sequence_length,
        )

    def __call__(self, prompt: str) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
        inputs = self.tokenize(prompt)
        kv_cache = self._initialize_kv_cache_state()

        onnxruntime_inputs = dict(
            attention_mask=inputs["attention_mask"],
            input_ids=inputs["input_ids"],
            **kv_cache,
        )

        if "positions" in self.model_inputs:
            attention_mask = inputs["attention_mask"]
            positions = attention_mask.cumsum(1) * attention_mask - 1
            onnxruntime_inputs["positions"] = positions

        if "causal_mask" in self.model_inputs:
            causal_mask = create_causal_mask(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            onnxruntime_inputs["causal_mask"] = causal_mask

        # run inference and return the cache state
        outputs = self.session.run(None, onnxruntime_inputs)
        prompt_logits, *prompt_cache = outputs

        # remove logits that correspond to padding tokens
        prompt_logits = numpy.compress(
            onnxruntime_inputs["attention_mask"].flatten(), prompt_logits, axis=1
        )  # (1, prompt_length, vocab_size)
        prompt_logits = prompt_logits[:, :-1, :]  # (1, prompt_length, vocab_size)

        # remove cache that corresponds to padding tokens
        prompt_cache = [
            numpy.compress(
                onnxruntime_inputs["attention_mask"].flatten(), cache, axis=2
            )
            for cache in prompt_cache
        ]  # List[(1, num_heads, past_length, head_dim)]

        return prompt_logits, prompt_cache

    def _initialize_kv_cache_state(self, length: int = 0) -> Dict[str, numpy.ndarray]:
        model = onnx.load(self.model_onnx_path, load_external_data=False)

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
            input_.name: numpy.zeros(
                (batch_size, num_attention_heads, length, hidden_dims),
                dtype=numpy.float32,
            )
            for input_ in model.graph.input
            if input_.name.startswith(CACHE_INPUT_PREFIX)
        }
        return kv_cache


class TorchGroundTruthSource(GroundTruthSource):
    """
    An object that generates ground truth logits and
    cache states from a prompt. This object can
    generate tokens in an autoregressive manner, and thus
    will output prompt logits, generated logits, generated
    sequence and prompt cache state
    """

    def __init__(self, num_tokens_to_generate: int, model_name: str):
        super().__init__(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.num_tokens_to_generate = num_tokens_to_generate

    def tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")

    def __call__(
        self, prompt: str
    ) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray], str]:
        # afaik it is not possible to get 'past_key_values' from
        # the generate method, so we have to run the model twice
        out = self.model.generate(
            self.tokenize(prompt).input_ids,
            max_new_tokens=self.num_tokens_to_generate,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        generated_text = self.tokenizer.decode(
            out.sequences[0], skip_special_tokens=True
        )
        generated_logits = numpy.concatenate(
            [[score.numpy() for score in out.scores]]
        ).transpose(
            1, 0, 2
        )  # (1, num_tokens_to_generate, vocab_size)

        out = self.model(**self.tokenize(prompt))
        prompt_logits = out.logits.detach().numpy()[
            :, :-1, :
        ]  # (1, prompt_length, vocab_size)
        prompt_cache = [
            entry.detach().numpy()
            for key_value_tuple in out.past_key_values
            for entry in key_value_tuple
        ]  # List[(1, num_heads, past_length, head_dim)]

        return generated_logits, prompt_logits, prompt_cache, generated_text
