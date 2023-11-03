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
"""
Integration of the `llm_evaluation_harness`:
https://github.com/EleutherAI/lm-evaluation-harness
"""

import json
import os

import numpy

import torch
from deepsparse import DEEPSPARSE_ENGINE, ORT_ENGINE, Pipeline
from lm_eval import evaluator, tasks, utils
from lm_eval.base import BaseLM
from src.deepsparse.eval.utils import initialize_model_from_target


def integration_eval(
    target,
    target_args,
    datasets,
    batch_size,
    engine_type,
    splits,
    metrics,
    engine_args,
    **kwargs,
):
    """
    Reimplementation of:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py
    that is compatible with our evaluation module
    """
    # [START]
    # The code that sets up the interface between deepsparse and llm_evaluation_harness
    if engine_type in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        model = DeepSparseLM(target, batch_size, **target_args)
    else:
        model = initialize_model_from_target(target, engine_type, **target_args)

    datasets = (",").join(datasets) if isinstance(datasets, list) else datasets

    # [END]

    # [START]
    # The code below is being adapted from:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py

    provide_description = kwargs.get("provide_description", False)
    limit = kwargs.get("limit", None)
    model_args = kwargs.get("model_args", "")
    num_fewshot = kwargs.get("num_fewshot", 0)
    max_batch_size = kwargs.get("max_batch_size", None)
    device = kwargs.get("device", None)
    no_cache = kwargs.get("no_cache", False)
    decontamination_ngrams_path = kwargs.get("decontamination_ngrams_path", None)
    check_integrity = kwargs.get("check_integrity", True)
    write_out = kwargs.get("write_out", True)
    output_base_path = kwargs.get("output_base_path", None)

    assert not provide_description  # not implemented

    if kwargs.get("limit"):
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. "
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if datasets is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(datasets.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if kwargs.get("description_dict_path"):
        with open(kwargs.get("description_dict_path"), "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_names,
        description_dict=description_dict,
        batch_size=batch_size,
        model_args=model_args,
        num_fewshot=num_fewshot,
        max_batch_size=max_batch_size,
        device=device,
        no_cache=no_cache,
        limit=limit,
        decontamination_ngrams_path=decontamination_ngrams_path,
        check_integrity=check_integrity,
        write_out=write_out,
        output_base_path=output_base_path,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    output_path = kwargs.get("output_path", None)
    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{model} ({model_args}), "
        f"limit: {limit}, "
        f"provide_description: {provide_description}, "
        f"num_fewshot: {num_fewshot}, "
        f"batch_size: {batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))
    # [END]

    # TODO: Add here the code to return the results in the format expected by the
    # evaluator module

    return results


class DeepSparseLM(BaseLM):
    # Default max sequence length setting for when no `max_length` is provided
    DEFAULT_MAX_LENGTH = 2048
    """
    A wrapper around the Deepsparse pipeline to make it compatible with the
    llm_evaluation_harness. DeepSparseLM is a subclass of BaseLM, uses the
    same interface as the other models in llm_evaluation_harness.

    :param target: The target to be evaluated
    :param target_args: The arguments for the target
    """

    def __init__(self, target: str, batch_size: int = 1, **kwargs):
        self.model = Pipeline.create(task="text_generation", model_path=target)
        self._max_length = kwargs.get("max_length", self.DEFAULT_MAX_LENGTH)
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length or self.default_max_length

    def tok_encode(self, string):
        return self.model.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.model.tokenizer.decode(tokens)

    def _model_call(self, inps) -> "torch.Tensor":
        """
        Override the _model_call method to use the
        Deepsparse pipeline for logits generation.

        :param inps: The input tokens passed from
            the llm_evaluation_harness
        :return: The torch tensor with logits for
            the input tokens. The shape of the logits
            tensor is (batch_size, seq_len, vocab_size)
        """
        # TODO: Enable batching the inps and then passing them
        # all at once to the pipeline for faster inference

        # encode the tokens to strings
        prompt = self.model.tokenizer.batch_decode(inps.numpy())

        # run the model to map the prompt to logits
        out = self.model(
            prompt=prompt,
            max_new_tokens=0,
            include_prompt_logits=True,
            output_scores=True,
        )
        logits_numpy = numpy.stack([generation.score for generation in out.generations])
        return torch.from_numpy(logits_numpy)

    def _model_generate(self, context, max_length, eos_token_id):
        # encode the tokens to strings
        prompt = self.model.tokenizer.batch_decode(context.numpy())
        out = self.model(
            prompt=prompt, max_new_tokens=max_length, force_max_tokens=True
        )
        return numpy.array(
            [self.model.tokenizer(prompt[0] + out.generations[0].text)["input_ids"]]
        )

    @property
    def device(self):
        pass

    @property
    def eot_token_id(self):
        pass

    @property
    def max_gen_toks(self):
        return 0
