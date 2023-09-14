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

from typing import List, Tuple

import numpy
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import snapshot_download


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

# SPARSEZOO_MODELS_TO_TEST should be a list of tuples in the form:
# [(model_stub, model_name, prompt, uses_bos_token, logits_treshold), ...],
# where
# - model_stub is the stub of the sparsezoo model to be tested
# - model_name is the name of the HuggingFace model that corresponds
#   to the sparsezoo model (required for ground truth generation)
# - prompt is the prompt to be used for testing
# - uses_bos_token is a boolean that indicates whether the tokenizer
#   prepends a bos token to the prompt during tokenization
# - logits_threshold is the treshold for the max difference between the
#   actual and the expected logits in the situations where they will not be
#   able to match the ground truth (e.g. when the DeepSparse pipeline is
#   running after the KV cache has been filled up). This value is established
#   empirically for each combination of prompt/pipeline/num_generated tokens.
SPARSEZOO_MODELS_TO_TEST = [
    (
        "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",  # noqa: E501
        "salesforce/codegen-350m-mono",
        CODE_LANGUAGE_PROMPT,
        False,
        13.0,
    ),
    (
        "zoo:nlg/text_generation/opt-1.3b/pytorch/huggingface/opt_pretrain/base-none",
        "facebook/opt-1.3b",
        NATURAL_LANGUAGE_PROMPT,
        True,
        3.9,
    ),
]
# TINYSTORIES_MODELS_TO_TEST works analogously to SPARSEZOO_MODELS_TO_TEST
# but it contains a lightweight model that can be used for testing without
# the need to download a large model from SparseZoo
TINYSTORIES_MODELS_TO_TEST = [
    (
        snapshot_download(repo_id="mgoin/TinyStories-1M-deepsparse"),
        "roneneldan/TinyStories-1M",
        NATURAL_LANGUAGE_PROMPT,
        False,
        29.6,
    )
]


def generate_pytest_params(
    use_sparsezoo_models: bool, cache_management_type: List[str]
):
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
    stubs_to_test = (
        SPARSEZOO_MODELS_TO_TEST if use_sparsezoo_models else TINYSTORIES_MODELS_TO_TEST
    )

    return cache_to_test, stubs_to_test


class TorchGroundTruthSource:
    """
    An object that generates ground truth logits and
    cache states from a prompt. This object can
    generate tokens in an autoregressive manner, and thus
    will output:
     - prompt logits,
     - generated logits,
     - prompt cache state,
     - generated sequence
    """

    def __init__(self, num_tokens_to_generate: int, model_name: str):

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = self._create_tokenizer(model_name)

        self.num_tokens_to_generate = num_tokens_to_generate
        self.model_name = model_name

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

    @staticmethod
    def _create_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer
