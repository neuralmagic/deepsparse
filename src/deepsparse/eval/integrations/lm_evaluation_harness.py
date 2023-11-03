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

from deepsparse import Pipeline
from lm_eval import evaluator
from lm_eval.base import BaseLM
from typing import Union


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


def integration_eval(
        target: Union[str, "Module"],
        target_args,
        datasets,
        splits,
        metrics,
        batch_size,
        engine,
        engine_args,
        **kwargs,
):

    if isinstance(target, str):
        target = DeepSparseLM(stub=target)
    else:
        pass # model is a torch.Module

    results = evaluator.simple_evaluate(
          model=kwargs.get("model", target),
          model_args=kwargs.get("model_args", target_args),
          tasks=kwargs.get("tasks", datasets),
    #     num_fewshot=num_fewshot,
    #     batch_size=batch_size,
    #     max_batch_size=max_batch_size,
    #     device=device,
    #     no_cache=no_cache,
    #     limit=limit,
    #     description_dict=description_dict,
    #     decontamination_ngrams_path=decontamination_ngrams_path,
    #     check_integrity=check_integrity,
    #     write_out=write_out,
    #     output_base_path=output_base_path,
          **kwargs,
    )


if __name__ == "__main__":
    target = "hf:mgoin/TinyStories-1M-deepsparse"
    datasets = ["hellaswag"]
    target_args = ""
    limit = 2 # testing purposes
    integration_eval(target=target, datasets=datasets, target_args=target_args, limit=limit, splits=None, metrics=None, batch_size=1, engine=None, engine_args=None)
