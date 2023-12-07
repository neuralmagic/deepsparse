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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
from pydantic import BaseModel, Field
from tqdm import tqdm

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
        model = DeepSparseLM(
            target=target, batch_size=batch_size, engine_type=engine_type
        )
    else:
        model = text_generation_model_from_target(target, engine_type)

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
        raw=dict(output=results_raw, input=filter_evaluator_input(evaluator_input)),
        formatted=format_raw_results(results_raw),
    )

    return results


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
        False, description="Include this flag to check integrity."
    )
    write_out: bool = Field(False, description="Include this flag to write out.")
    output_base_path: Optional[str] = Field(
        None, description="Specify the output base path."
    )


class DeepSparseLM(base.BaseLM):
    # Default max sequence length setting for when no `max_length` is provided
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        target: str,
        tokenizer: Optional[str] = None,
        engine_type: Union[ORT_ENGINE, DEEPSPARSE_ENGINE] = DEEPSPARSE_ENGINE,
        batch_size: Optional[Union[int, str]] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        """
        Wrapper around the DeepSparse pipeline to make it compatible with the
        llm-evaluation-harness.
        """
        super().__init__()

        self._batch_size = int(batch_size)
        self._max_length = max_length or self._DEFAULT_MAX_LENGTH
        self._max_gen_toks = max_gen_toks

        # Initialize new model and tokenizer instances
        self.model = Pipeline.create(
            task="text-generation",
            model_path=target,
            sequence_length=self._max_length,
            trust_remote_code=trust_remote_code,
            engine_type=engine_type,
            # should pass batch_size but the
            # TextGeneration model does not support batch_size > 1
            batch_size=1,
        )
        self.tokenizer = tokenizer if tokenizer else self.model.tokenizer

        self.vocab_size = self.tokenizer.vocab_size

    def _model_call(self, inps) -> torch.Tensor:
        """
        Override the _model_call method to use the DeepSparse pipeline for
        logits generation.

        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        # Encode the tokens to strings
        prompt = self.model.tokenizer.batch_decode(inps.numpy())

        # Run the model to map the prompt to logits
        out = self.model(
            prompt=prompt,
            max_new_tokens=0,
            include_prompt_logits=True,
            output_scores=True,
        )
        logits_numpy = numpy.stack([generation.score for generation in out.generations])
        return torch.from_numpy(logits_numpy)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length, int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            responses = self.model(
                sequences=context,
                max_new_tokens=max_tokens,
                stop=until,
                do_sample=False,
            )

            responses = responses if type(responses) is list else [responses]

            for response in responses:
                response = response.generations[0].text
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), response)
                results.append(response)

        return reorder.get_original(results)

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        # should return self._batch_size but the
        # TextGeneration model does not support batch_size > 1
        return 1

    @property
    def device(self):
        pass

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)
