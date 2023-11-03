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

from typing import Optional

from transformers import AutoModelForCausalLM

from deepsparse import Pipeline
from lm_eval import evaluator
from lm_eval.base import BaseLM


def integration_eval(
    target,
    target_args,
    datasets,
    batch_size,
    splits=None,
    metrics=None,
    engine_type=None,
    engine_args=None,
    **kwargs,
):
    model = initialize_model(target, target_args)

    evaluator.simple_evaluate(
        model=model,
        model_args=kwargs.get("model_args", target_args),
        tasks=kwargs.get("tasks", datasets),
        batch_size=batch_size,
        **kwargs,
    )

    return True


class DeepSparseLM(BaseLM):
    # potentially create pipeline inside of this class
    def __init__(self, stub, max_length: Optional[int] = None):
        self.model = Pipeline.create(task="text_generation", model_path=stub)

        self.default_max_length = 1024
        self._max_length = max_length

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline):
        return cls(pipeline)

    @property
    def batch_size(self):
        return self.model._batch_size

    @property
    def eot_token(self) -> str:
        pass

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    def _loglikelihood_tokens(self, requests, **kwargs):
        pass

    @property
    def device(self):
        return "cpu"

    @property
    def eot_token_id(self) -> int:
        pass

    @property
    def max_gen_toks(self):
        pass

    @property
    def max_length(self):
        return self._max_length or self.default_max_length

    def tok_encode(self, string: str):
        return self.model.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)


def initialize_model(target, target_args):
    # creates model: Union[DeepSparseLM, Module]
    # given the target
    return AutoModelForCausalLM.from_pretrained(target)
