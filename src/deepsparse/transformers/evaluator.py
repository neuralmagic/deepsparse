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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from deepsparse.pipeline import DEEPSPARSE_ENGINE
from deepsparse.transformers.eval_downstream import perplexity_eval


_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


@dataclass
class Evaluator(ABC):
    """
    Evaluator base class that represents the contract all evaluators must follow.

    :param model: The path to the model to evaluate.
    :param dataset: The name of the dataset to evaluate on.
    :param eval: The list of evaluation metrics to run.
    """

    def __init__(self, model: str, dataset: str, eval: Optional[List[str]] = None):
        self.model_path_ = model
        self.dataset_name_ = dataset
        self.eval_ = eval
        self.results_ = None

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Main method for running the evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_results(self):
        """
        Return the results of the evaluation.
        """
        raise NotImplementedError


class LMEvaluator(Evaluator):
    """
    Evaluator specific to transformers.

    :param model: The path to the model to evaluate. Must be a directory
        containing `model.onnx` with configuration files, or a SparseZoo stub.
    :param dataset: The name of the dataset to evaluate on. Supported datasets
        for Language Models are `openai_humaneval`, `wikitext2` and `c4`.
    :param eval: The list of evaluation metrics to run. Supported metrics for
        Language Models are `perplexity`.
    """

    VALID_DATSETS: List[str] = ["openai_humaneval", "wikitext2", "c4"]

    def __init__(self, model: str, dataset: str, eval: Optional[List[str]] = None):
        super().__init__(model=model, dataset=dataset, eval=eval)
        self.eval_ = self.eval_ or ["perplexity"]
        if self.eval_ != ["perplexity"]:
            raise ValueError(
                "Only `perplexity` is supported for "
                f"Language Models but got {self.eval_}"
            )
        if self.dataset_name_ not in self.VALID_DATSETS:
            raise ValueError(
                f"Only {self.VALID_DATSETS} are supported for "
                f"Language Models but got {self.dataset_name_}"
            )

    def evaluate(self, **kwargs):
        """
        Evaluate the model on the given dataset and populates results

        :param kwargs: Additional arguments to pass to the evaluation function.
        """
        # unique eval metrics
        eval_ = list(set(self.eval_))

        for metric in eval_:
            _LOGGER.debug(f"Evaluating {metric}")
            if metric == "perplexity":
                self.results_ = self._calc_perplexity(**kwargs)

    def get_results(self) -> Dict[str, Any]:
        """
        Return the results of the evaluation.
        """
        if not self.results_:
            raise RuntimeError("No results found. Run `evaluate` first.")

        return self.results_.compute()

    def _calc_perplexity(self, **kwargs):
        """
        Calculate the perplexity of the model.
        
        Note: Right now, this function relies on the `perplexity_eval`
        function from `deepsparse.transformers.eval_downstream` to calculate
        perplexity. This is subject to change in the future.
        

        :param kwargs: Additional arguments to pass to the
            perplexity function.
        """

        # default arguments for perplexity_eval
        #  see src/deepsparse/transformers/eval_downstream.py for
        #  full list of arguments, the default values can be overridden
        #  by passing them as keyword arguments to this function

        default_arguments = dict(
            model_path=self.model_path_,
            max_sequence_length=384,
            engine=DEEPSPARSE_ENGINE,
            num_cores=0,
            trust_remote_code=True,
            max_samples=None,
            batch_size=1,
            kwargs=None,
        )

        # wrap arguments in SimpleNamespace to allow dot access
        #  as expected by `perplexity_eval``
        perplexity_args = SimpleNamespace(**default_arguments, **kwargs)

        # dataset_name must be passed as a keyword argument
        perplexity_kwargs = dict(dataset_name=self.dataset_name_)

        return perplexity_eval(perplexity_args, **perplexity_kwargs)
