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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy
from tqdm import tqdm

from datasets import load_dataset
from deepsparse import Pipeline
from deepsparse.evaluation.registry import EvaluationRegistry
from deepsparse.evaluation.results import Dataset, Evaluation, Metric, Result
from deepsparse.evaluation.utils import PERPLEXITY
from deepsparse.transformers.metrics import Perplexity
from deepsparse.transformers.pipelines.text_generation import TextGenerationPipeline
from deepsparse.transformers.pipelines.text_generation.pipeline_no_kv_cache import (
    TextGenerationPipelineNoCache,
)
from deepsparse.transformers.utils.eval_helpers import (
    HumanEvalIteratorWrapper,
    process_concatenated_datasets,
)


"""
Integration for the evaluation module
that computes the perplexity of a model on a dataset
"""
_LOGGER = logging.getLogger(__name__)


@EvaluationRegistry.register(name=PERPLEXITY)
def integration_eval(
    pipeline: Pipeline,
    datasets: Union[List[str], str] = "openai_humaneval",
    batch_size: int = 1,
    limit: Optional[int] = None,
    accumulate: Optional[bool] = None,
    splits: Union[List[str], str, None] = "test",
    metrics: Union[List[str], str, None] = None,
    **kwargs,
) -> Result:
    """
    A function that computes the perplexity of a pipeline given a set
    of dataset names.

    :param pipeline: the pipeline to evaluate. The assumed pipeline
        is a TextGenerationPipeline, either with or without the KV
        cache support
    :param datasets: the names of dataset(s) to evaluate on
    :param batch_size: the batch size to use for evaluation
    :param splits: the split of the dataset to evaluate on. Default is "test"
    :param metrics: the metrics to compute. Default is None
    :param limit: the number of batches to evaluate on. Default is None
        (evaluates on entire dataset)
    :param accumulate: whether to perplexity computation should
        accumulate negative log-likelihood over samples. Defaults to
        the default accumulate variable inferred from the dataset in
        `datasets`. If not None, it will override the inferred accumulate
         variable.
    :return: a Result object containing the raw and formatted results
    """
    metrics = metrics or PERPLEXITY
    if metrics != PERPLEXITY:
        raise ValueError(f"Invalid metric {metrics} for perplexity evaluation")
    if splits is None:
        splits = "test"
        _LOGGER.info("Argument `splits` is None. Defaulting to `test` split.")
    datasets = datasets if isinstance(datasets, list) else [datasets]
    results_raw = defaultdict(str)
    for dataset_name in datasets:
        results_raw[dataset_name] = defaultdict()
        dataset, _accumulate = load_perplexity_dataset(
            dataset_name=dataset_name, splits=splits, pipeline=pipeline, **kwargs
        )
        if accumulate is None:
            accumulate = _accumulate
        else:
            _LOGGER.info(
                f"Argument `accumulate` set to {accumulate}. "
                "Overriding the inferred accumulate variable from the dataset."
            )

        perplexity = run_perplexity(
            pipeline=pipeline,
            dataset=dataset,
            batch_size=batch_size,
            accumulate=accumulate,
            limit=limit,
        )

        results_raw[dataset_name] = defaultdict()
        results_raw[dataset_name]["results"] = perplexity
        results_raw[dataset_name]["split"] = splits

    results = Result(
        # omit storing raw results. they can potentially
        # contain numpy arrays that are not serializable.
        # all the information is stored in the formatted results
        raw=None,
        formatted=format_raw_results(results_raw),
    )

    return results


def run_perplexity(
    pipeline: Union[TextGenerationPipelineNoCache, TextGenerationPipeline],
    dataset: "Dataset",
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
    for idx, sample in _enumerate_progress(
        dataset, max_steps=None if limit is None else limit * batch_size
    ):

        if limit is not None:
            # stop if we have reached the #limit
            # number of batches to be processed
            if idx >= limit * batch_size:
                break

        batch.append(sample)

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

            for s in range(batch_size):
                # Need to remove tokens that were masked
                input_ids = out.input_tokens["input_ids"][s].flatten()
                attention_mask = out.input_tokens["attention_mask"][s].flatten()
                logits = out.generations[s].score
                if batch_size > 1 and isinstance(
                    pipeline, TextGenerationPipelineNoCache
                ):
                    logits = logits[-attention_mask.sum() :, :]

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


def load_perplexity_dataset(
    dataset_name: str,
    splits: Union[List[str], str] = "test",
    pipeline: Optional[Pipeline] = None,
    **kwargs,
):
    """
    Function to load the dataset for perplexity computation.
    Eventually we want to load the dataset from the nm_utils

    :param dataset_name: the name of the dataset to load
    :param splits: the splits to load from the dataset. Default is "test"
    :param pipeline: the pipeline to use for loading the dataset. The pipeline
        is used to infer the model path and sequence length to use for loading
        the dataset. This argument can be omitted if the appropriate kwargs
        are provided, or if the dataset does not require a process_concatenated_datasets
        function to load the dataset.
    :param kwargs: additional keyword arguments to pass to the dataset loading function
    :return: the dataset and whether to accumulate perplexity over samples
    """
    if isinstance(splits, list):
        raise NotImplementedError("Evaluation on multiple splits not implemented")

    if dataset_name == "openai_humaneval":
        dataset = load_dataset(dataset_name, split=splits)
        dataset = HumanEvalIteratorWrapper(dataset)
        accumulate = False
    elif dataset_name in {"wikitext2", "c4"}:
        # fetch max_sequence_length from pipeline if not provided
        max_sequence_length = kwargs.pop("max_sequence_length", None)
        if max_sequence_length is None and pipeline is not None:
            max_sequence_length = pipeline.sequence_length

        # fetch model_path from pipeline if not provided
        model_path = kwargs.pop("model_path", None)
        if model_path is None and pipeline is not None:
            model_path = os.path.dirname(pipeline.model_path)

        dataset = process_concatenated_datasets(
            dataset_name,
            model_path=model_path,
            max_sequence_length=max_sequence_length,
            split=splits,
            **kwargs,
        )
        accumulate = True
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    return dataset, accumulate


def _enumerate_progress(dataset, max_steps):
    progress_bar = tqdm(dataset, total=max_steps) if max_steps else tqdm(dataset)
    return enumerate(progress_bar)
