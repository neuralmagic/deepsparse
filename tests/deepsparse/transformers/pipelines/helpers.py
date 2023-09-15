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
from typing import List, Tuple

import numpy
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import snapshot_download


def generate_pytest_params(config_files_dir: str, cadence_to_enable=["commit"]):
    assert os.path.isdir(config_files_dir)
    test_params = []
    run_main_tests_only = []
    for file in os.listdir(config_files_dir):
        with open(os.path.join(config_files_dir, file), "r") as f:
            config = yaml.safe_load(f)
            if config["cadence"] in cadence_to_enable:
                test_params.append(_process_main_test_params(config))
                run_main_tests_only.append(config["run_main_tests_only"])
    return (test_params, run_main_tests_only)


def _process_main_test_params(config):
    model_path = config["model_path"]
    if not model_path.startswith("zoo:"):
        model_path = snapshot_download(repo_id=model_path)
    return (
        model_path,
        config["model_name"],
        config["num_tokens_generate"],
        config["prompt"],
        config["has_bos_token"],
        config["logits_threshold"],
        config["precision"],
        _validate_cache_management_type(config["cache_management_type"]),
    )


def _validate_cache_management_type(cache_management_type):
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
    return cache_management_type


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
