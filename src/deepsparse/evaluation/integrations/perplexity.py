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

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy
from tqdm import tqdm

from datasets import load_dataset
from deepsparse.evaluation.registry import EvaluationRegistry
from deepsparse.evaluation.results import Dataset, Evaluation, Metric, Result
from deepsparse.transformers.metrics import Perplexity
from deepsparse.transformers.pipelines.text_generation import TextGenerationPipeline
from deepsparse.transformers.pipelines.text_generation.pipeline_no_kv_cache import (
    TextGenerationPipelineNoCache,
)


"""
Integration for the evaluation module
that computes the perplexity of a model on a dataset
"""


@EvaluationRegistry.register(name="perplexity")
def integration_eval(
    pipeline: Union[TextGenerationPipelineNoCache, TextGenerationPipeline],
    datasets: Union[List[str], str],
    batch_size: int = 1,
    split: str = "test",
    limit: Optional[int] = None,
) -> Result:
    """
    A function that computes the perplexity of a pipeline given a set
    of dataset names.

    :param pipeline: the pipeline to evaluate. The assumed pipeline
        is a TextGenerationPipeline, either with or without the KV
        cache support
    :param datasets: the names of dataset(s) to evaluate on
    :param batch_size: the batch size to use for evaluation
    :param split: the split of the dataset to evaluate on. Default is "test"
    :param limit: the number of batches to evaluate on. Default is None
        (evaluates on entire dataset)
    :return: a Result object containing the raw and formatted results
    """
    datasets = datasets if isinstance(datasets, list) else [datasets]
    results_raw = defaultdict(str)
    for dataset_name in datasets:
        results_raw[dataset_name] = defaultdict()
        dataset, accumulate = load_perplexity_dataset(dataset_name, split)
        perplexity = run_perplexity(
            pipeline=pipeline,
            dataset=dataset,
            batch_size=batch_size,
            accumulate=accumulate,
            limit=limit,
        )

        results_raw[dataset_name] = defaultdict()
        results_raw[dataset_name]["results"] = perplexity
        results_raw[dataset_name]["split"] = split

    results = Result(
        raw=results_raw,
        formatted=format_raw_results(results_raw),
    )

    return results


def run_perplexity(
    pipeline: Union[TextGenerationPipelineNoCache, TextGenerationPipeline],
    dataset: Any,  # TODO: Edit, once we agree on the dataset registry
    batch_size: int,
    accumulate: bool,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute the perplexity of a pipeline given a dataset.

    :param pipeline: the pipeline to evaluate. The assumed pipeline
        is a TextGenerationPipeline, either with or without the KV
        cache support
    :param dataset: the dataset to evaluate on
    :param batch_size: the batch size to use for evaluation
    :param accumulate: whether to perplexity computation should
        accumulate negative log-likelihood over samples
    :param limit: the number of batches to evaluate on. Default is None
        (evaluates on entire dataset)

    :return: a dictionary containing the perplexity results
    """

    perplexity = Perplexity(accumulate=accumulate)

    batch = []
    for idx, sample in enumerate(tqdm(dataset)):

        # TODO: To remove when we import the dataset registry
        sample = sample["prompt"] + sample["canonical_solution"]

        if limit is not None:
            # stop if we have reached the #limit
            # number of batches to be processed
            if idx >= limit * batch_size:
                break

        batch.append(sample)

        # TODO: Assert here that the pipeline has set
        # include_prompt_logits=True,
        # return_input_tokens=True,

        if len(batch) == batch_size:
            if isinstance(pipeline, TextGenerationPipelineNoCache):
                out = pipeline(
                    prompt=batch,
                    output_scores=True,
                    include_prompt_logits=True,
                    return_input_tokens=True,
                )
            else:
                out = pipeline(
                    prompt=batch,
                    output_scores=True,
                    max_new_tokens=0,
                    include_prompt_logits=True,
                    return_input_tokens=True,
                )

            # TODO: Perhaps we can vectorize it to be more efficient
            # and elegant
            for s in range(batch_size):
                # Need to remove tokens that were masked
                input_ids = out.input_tokens["input_ids"][s].flatten()
                attention_mask = out.input_tokens["attention_mask"][s].flatten()
                logits = out.generations[s].score

                logits = numpy.compress(attention_mask, logits, axis=0)[:-1, :]
                input_ids = numpy.compress(attention_mask, input_ids)[1:]

                # Add predictions (logits) and targets (input_ids) to metric
                perplexity.add_batch(logits, input_ids)

            batch.clear()

    return perplexity.compute()


def format_raw_results(results: Dict[str, Any]) -> List[Evaluation]:
    """
    Format the raw perplexity results into a list of
    Evaluation objects.

    :param results: the raw results from perplexity computation
    :return: the formatted results as a list of Evaluation objects
    """
    formatted_results = []
    for dataset_name, dataset_result in results.items():
        metrics = []
        for metric_name, metric_value in dataset_result["results"].items():
            if isinstance(metric_value, numpy.ndarray):
                metric_value = metric_value.tolist()
            metric = Metric(name=metric_name, value=metric_value)
            metrics.append(metric)
        dataset = Dataset(type=None, name=dataset_name, split=dataset_result["split"])
        evaluation = Evaluation(
            task="perplexity",
            dataset=dataset,
            metrics=metrics,
            samples=None,
        )
        formatted_results.append(evaluation)
    return formatted_results


def load_perplexity_dataset(dataset_name: str, split="test"):
    """
    Dummy function to load the dataset for perplexity computation.
    Eventually we want to load the dataset from the nm_utils
    """
    if dataset_name == "openai_humaneval":
        dataset = load_dataset(dataset_name, split=split)
        accumulate = False
        return dataset, accumulate
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
