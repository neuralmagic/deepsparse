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

import logging
import os
from typing import Any, Dict, List, Tuple, Union

import numpy
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import pytest


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


def parse_params(configs_directory: str) -> List[Dict[str, Any]]:
    # parses the config file provided
    assert os.path.isdir(
        configs_directory
    ), f"Config_directory {configs_directory} is not a directory"

    config_dicts = []
    for file in os.listdir(configs_directory):
        if file.endswith(".yaml"):
            config_path = os.path.join(configs_directory, file)
            # reads the yaml file
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            cadence = os.environ.get("CADENCE", "commit")
            expected_cadence = config["cadence"]

            if not isinstance(expected_cadence, list):
                expected_cadence = [expected_cadence]
            if cadence in expected_cadence:
                config_dicts.append(config)
            else:
                logging.info(
                    f"Skipping testing model: {config['model_path']} "
                    f"for cadence: {config['cadence']}"
                )
        else:
            raise FileNotFoundError(
                f"Could not find a yaml file in {configs_directory}"
            )
    return config_dicts


def validate_internal_kv_cache(
    internal_kv_cache, available_kv_cache_types: Union[str, List[str]]
) -> bool:
    if internal_kv_cache and True not in available_kv_cache_types:
        pytest.skip(
            "The tests for running the pipeline with "
            "internal kv cache management are disabled."
        )
    if not internal_kv_cache and False not in available_kv_cache_types:
        pytest.skip(
            "The tests for running the pipeline with "
            "external kv cache management are disabled."
        )
    return internal_kv_cache


def validate_task(task: str, available_tasks: Union[str, List[str]]) -> bool:
    if task not in available_tasks:
        pytest.skip(
            f"The tests for running the pipeline with task: {task} are disabled. "
            f"The available tasks, as specified in the config are: {available_tasks}"
        )
    return task
