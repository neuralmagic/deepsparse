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
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
from tqdm import tqdm
from transformers import AutoTokenizer
import copy
from deepsparse import Pipeline
from deepsparse.evaluation.registry import EvaluationRegistry
from deepsparse.evaluation.results import Dataset, Evaluation, Metric, Result
from deepsparse.utils.data import numpy_log_softmax
from lm_eval import evaluator, tasks, utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from collections import defaultdict


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
        model = DeepSparseLM(pipeline=model, batch_size=batch_size)

    datasets = (",").join(datasets) if isinstance(datasets, list) else datasets
    task_names = utils.pattern_match(datasets.split(","), tasks.ALL_TASKS)

    _LOGGER.info(f"Selected Tasks: {task_names}")

    results_raw = evaluator.simple_evaluate(
        model=model, tasks=task_names, batch_size=batch_size, **kwargs
    )

    results = Result(
        raw=results_raw,
        formatted=format_raw_results(results_raw),
    )

    return results


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
            if isinstance(metric_value, str):
                continue
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
        batch_size: int = 1,
        max_gen_toks: int = 128,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Wrapper around the DeepSparse pipeline to make it compatible with the
        llm-evaluation-harness.
        """
        super().__init__()

        # Initialize new model and tokenizer instances
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.tokenizer = tokenizer or pipeline.tokenizer
        self._max_length = pipeline.sequence_length
        self._max_gen_toks = max_gen_toks

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Copied directly from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                raise NotImplemented("Implementing empty context is not supported yet")
            context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            """Defines the key for the sorted method"""
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for chunk in tqdm(
            list(utils.chunks(re_ord.get_reordered(), self.batch_size)),
            disable=disable_tqdm,
        ):
            for cache_key, context_enc, continuation_enc in chunk:
                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                inp = (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]

                response = self.pipeline(
                    prompt=self.tokenizer.decode(inp),
                    max_new_tokens=0,
                    output_scores=True,
                    include_prompt_logits=True,
                )

                for i, resp in enumerate(response.generations):
                    # (seq_len, vocab_size)
                    multi_scores = resp.score
                    # (seq_len, vocab_size) but with softmax applied
                    multi_logits = numpy_log_softmax(multi_scores, axis=1)
                    # toss out the context half of the sequence
                    # (cont_len, vocab_size)
                    continuation_multi_logits = multi_logits[-len(continuation_enc) :]

                    # pick out the logits for the continuation tokens
                    # (cont_len,)
                    continuation_logits = continuation_multi_logits[
                        numpy.arange(len(continuation_enc)), continuation_enc
                    ]
                    # check if the tokens generated greedly are the same
                    # as the expected continuation
                    greedy_tokens = continuation_multi_logits.argmax(axis=1)
                    max_equal = greedy_tokens.tolist() == continuation_enc

                    # Answer: (log prob, is-exact-match)
                    answer = (float(continuation_logits.sum()), bool(max_equal))

                    res.append(answer)

                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        raise NotImplementedError()

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            # the negative sign on len(toks) sorts descending
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        pbar = tqdm(total=len(requests))
        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            chunks = utils.chunks(re_ord.get_reordered(), n=self.batch_size)
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [kwargs]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                    )

                if not until:
                    until = [self.tok_decode(self.eot_token_id)]

                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks

                # we require users to pass do_sample=True explicitly for non-greedy gen
                if "do_sample" not in kwargs.keys():
                    kwargs["do_sample"] = False

                # first stop sequence is used to halt generation upon encountering
                primary_until = [until[0]]

                responses = self.pipeline(
                    sequences=contexts,
                    max_new_tokens=max_gen_toks,
                    stop=until,
                    **kwargs,
                )

                responses = responses if type(responses) is list else [responses]
                for response, context in zip(responses, contexts):
                    text = response.generations[0].text
                    # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                    for term in until:
                        if len(term) > 0:
                            # ignore possible empty separators
                            text = text.split(term)[0]

                    res[key].append(text)
                    self.cache_hook.add_partial("greedy_until", (context, gen_kwargs), text)
                    pbar.update(1)
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        """
        Copied directly from
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc
