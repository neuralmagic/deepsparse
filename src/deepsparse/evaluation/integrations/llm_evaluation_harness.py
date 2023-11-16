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
import logging
from typing import Any, Dict, List, Optional, Union

import numpy
from pydantic import BaseModel, Field

import torch
from lm_eval import base, evaluator, tasks, utils
from src.deepsparse import DEEPSPARSE_ENGINE, ORT_ENGINE, Pipeline
from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.evaluation.results import Dataset, Evaluation, Metric, Result
from src.deepsparse.evaluation.utils import text_generation_model_from_target


_LOGGER = logging.getLogger(__name__)


@EvaluationRegistry.register(name=["llm_evaluation_harness", "llm-evaluation-harness"])
def integration_eval(
    target: str,
    datasets: Union[List[str], str],
    batch_size: int,
    engine_type: Optional[str] = None,
    **kwargs,
) -> Result:
    """
    Reimplementation of:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py
    that is compatible with deepsparse.evaluator.py

    :param target: the model name
    :param datasets: the datasets to evaluate on
    :param batch_size: the batch size to use for evaluation
    :param engine_type: the engine type to use for evaluation
    :param kwargs: additional arguments to alter the behavior of the evaluation

    :return the evaluation results
    """
    # [START]
    # The code that sets up the interface between deepsparse and llm_evaluation_harness
    if engine_type in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        model = DeepSparseLM(target, batch_size, **kwargs)
    else:
        model = text_generation_model_from_target(target, engine_type, **kwargs)

    datasets = (",").join(datasets) if isinstance(datasets, list) else datasets
    # [END]

    # [START]
    # The code below is being adapted from:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py
    if kwargs.get("limit"):
        _LOGGER.warning(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. "
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if datasets is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(datasets.split(","), tasks.ALL_TASKS)

    _LOGGER.info(f"Selected Tasks: {task_names}")

    description_dict = {}
    if kwargs.get("description_dict_path"):
        with open(kwargs.get("description_dict_path"), "r") as f:
            description_dict = json.load(f)

    evaluator_input = EvaluatorInputSchema(
        model=model,
        tasks=task_names,
        description_dict=description_dict,
        batch_size=batch_size,
        **kwargs,
    )

    results_raw = evaluator.simple_evaluate(**evaluator_input.dict())

    results = Result(
        raw=dict(output=results_raw, input=evaluator_input),
        formatted=format_raw_results(results_raw),
    )

    return results


def format_raw_results(results: Dict[str, Any]) -> List[Evaluation]:
    """
    Format the raw results from llm_evaluation_harness into a list of
    Evaluation objects.

    :param results: the raw results from llm_evaluation_harness
    :return: the formatted results as a list of Evaluation objects
    """
    formatted_results = []
    for dataset_name, dataset_result in results["results"].items():
        metrics = []
        for metric_name, metric_value in dataset_result.items():
            metric = Metric(name=metric_name, value=metric_value)
            metrics.append(metric)
        dataset = Dataset(
            type=None, name=dataset_name, config=results["config"], split=None
        )
        evaluation = Evaluation(
            task="llm_evaluation_harness",
            dataset=dataset,
            metrics=metrics,
            samples=None,
        )
        formatted_results.append(evaluation)
    return formatted_results


class EvaluatorInputSchema(BaseModel):
    model: Any = Field(description="The name of the model.")
    tasks: List[str] = Field(
        description="The task (or multiple tasks) to evaluate the target on."
    )
    description_dict: Optional[Dict[str, Any]] = Field(
        None, description="Description dict."
    )
    batch_size: int = Field(description="The batch size to use for evaluation.")
    model_args: str = Field(
        "", description="Additional arguments for the evaluated model."
    )
    num_fewshot: int = Field(0, description="The number of few shots to use.")
    max_batch_size: Optional[int] = Field(
        None, description="Maximal batch size to try with --batch_size auto."
    )
    device: Optional[str] = Field(None, description="Device to use for evaluation.")
    no_cache: bool = Field(False, description="Include this flag to prevent caching.")
    limit: Optional[float] = Field(
        None,
        description="Limit the number of examples per task. If <1, "
        "limit is a percentage of the total number of "
        "examples.",
    )
    decontamination_ngrams_path: Optional[str] = Field(
        None, description="Specify the path for decontamination n-grams."
    )
    check_integrity: bool = Field(
        True, description="Include this flag to check integrity."
    )
    write_out: bool = Field(False, description="Include this flag to write out.")
    output_base_path: Optional[str] = Field(
        None, description="Specify the output base path."
    )


class DeepSparseLM(base.BaseLM):
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
        return self._max_length

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
        # encode the prompt tokens to strings
        prompt = self.model.tokenizer.batch_decode(context.numpy())

        # run generation
        out = self.model(
            prompt=prompt, max_new_tokens=max_length, force_max_tokens=True
        )
        # return tokens for prompt + generated text
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
