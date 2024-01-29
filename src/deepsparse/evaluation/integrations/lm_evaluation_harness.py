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
Integration of the `lm_evaluation_harness`:
https://github.com/EleutherAI/lm-evaluation-harness
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
from pydantic import BaseModel, Field
from tqdm import tqdm

import torch
from deepsparse import Pipeline
from deepsparse.evaluation.registry import EvaluationRegistry
from deepsparse.evaluation.results import Dataset, Evaluation, Metric, Result
from deepsparse.transformers.metrics import _cross_entropy
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval import evaluator, utils, tasks
from lm_eval.__main__ import cli_evaluate
tasks.initialize_tasks("INFO")

_LOGGER = logging.getLogger(__name__)

__all__ = ["integration_eval"]


@EvaluationRegistry.register(name="lm-evaluation-harness")
def integration_eval(
    model: Any,
    datasets: Union[List[str], str],
    batch_size: int,
    **kwargs,
) -> Result:
    """
    Reimplementation of:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py
    that is compatible with deepsparse.evaluator.py

    :param model: the model/pipeline to evaluate
    :param datasets: the datasets to evaluate on
    :param batch_size: the batch size to use for evaluation
    :param kwargs: additional arguments to alter the behavior of the evaluation

    :return the evaluation results
    """
    if isinstance(model, Pipeline):
        model = DeepSparseLM(pipeline=model)


    datasets = (",").join(datasets) if isinstance(datasets, list) else datasets
    task_names = utils.pattern_match(datasets.split(","), tasks.ALL_TASKS)

    _LOGGER.info(f"Selected Tasks: {task_names}")

    results_raw = evaluator.simple_evaluate(model=model, tasks=task_names, batch_size=batch_size, **kwargs)

    # results = Result(
    #     raw=dict(output=results_raw, input=None),
    #     formatted=None,
    # )

    return results_raw


def filter_evaluator_input(
    evaluator_input: "EvaluatorInputSchema",
) -> Dict[str, Any]:  # noqa: F821
    """
    Filter the evaluator input to remove the model field.
    The model field is a complex object that cannot be serialized.

    :param evaluator_input: the evaluator input to filter
    :return: the filtered evaluator input
    """
    evaluator = evaluator_input.dict()
    del evaluator["model"]

    return evaluator


def format_raw_results(results: Dict[str, Any]) -> List[Evaluation]:
    """
    Format the raw results from lm_evaluation_harness into a list of
    Evaluation objects.

    :param results: the raw results from lm_evaluation_harness
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
            task="lm_evaluation_harness",
            dataset=dataset,
            metrics=metrics,
            samples=None,
        )
        formatted_results.append(evaluation)
    return formatted_results


class DeepSparseLM(LM):
    def __init__(
        self,
        pipeline: Pipeline,

    ):
        """
        Wrapper around the DeepSparse pipeline to make it compatible with the
        llm-evaluation-harness.
        """
        super().__init__()

        # Initialize new model and tokenizer instances
        self.pipeline = pipeline


    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        greedy = not self.pipeline.config.do_sample
        prompts = [request.arguments[0] for request in requests]
        out = self.pipeline(prompt = prompts,
                            output_scores=True,
                            )

        likelyhoods = []
        for prompt_idx, prompt in enumerate(prompts):
            logits = out.generations[prompt_idx].score
            tokenized_prompt = self.pipeline.tokenizer(prompt)
            nll = _cross_entropy(logits[:sum(tokenized_prompt["attention_mask"]),:], tokenized_prompt["input_ids"])
            likelyhoods.append((nll, greedy))
        return likelyhoods


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        pass

    def generate_until(self, requests: list[Instance]) -> list[str]:
        pass
