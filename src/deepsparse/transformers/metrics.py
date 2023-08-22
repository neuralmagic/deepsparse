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
Utilities for evaluation metric computation
"""

from typing import Any, Dict, Optional

import numpy

import torch
from sklearn.metrics import precision_recall_fscore_support


__all__ = [
    "PrecisionRecallF1",
    "Perplexity",
]


class Perplexity:
    def __init__(self, accumulate_likelihood: bool = False):
        """
        Given the pipeline, compute the perplexity of the model
        on the given text input.

        Code adapted from:
        https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py # noqa: E501

         non-overlapping batches
        """
        self._predictions = None
        self._targets = None
        self._loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        self._accumulate_likelihood = accumulate_likelihood

    def add_batch(self, predictions: numpy.ndarray, targets: numpy.ndarray):
        """
        adds a batch of prediction results to track, should be of shape
        (batch_size, num_labels)

        :param predictions: predicted scores from pipeline
        :param targets: target values - label column should be 1 if a label is positive
            0 otherwise
        """
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, predictions.shape[0])
        if targets.ndim == 1:
            targets = targets.reshape(1, targets.shape[0])

        if self._predictions is None:
            self._predictions = [predictions]
            self._targets = [targets]
        else:
            self._predictions.append(predictions)
            self._targets.append(targets)

    def compute(self) -> Dict[str, Any]:
        """
        :return: A dictionary containing the mean perplexity
            and the list of perplexities
        """
        # compile results into required str -> float dict
        neg_log_likelihoods = []
        for prediction, target in zip(self._predictions, self._targets):
            neg_log_likelihoods.append(
                self._loss_fct(
                    torch.tensor(prediction.transpose(0, 2, 1)),
                    torch.tensor(target),
                ).mean().item()
            )

        if self._accumulate_likelihood:
            neg_log_likelihood = numpy.mean(neg_log_likelihoods)
            return {"perplexity": numpy.exp(neg_log_likelihood)}
        else:
            perplexities = [numpy.exp(nll) for nll in neg_log_likelihoods]
            return {
                "perplexities": perplexities,
                "mean_perplexity": numpy.mean(perplexities),
            }


class PrecisionRecallF1:
    def __init__(self, id_to_label: Optional[Dict[int, str]] = None):
        self._id_to_label = id_to_label
        self._predictions = None
        self._targets = None

    def add_batch(self, predictions: numpy.ndarray, targets: numpy.ndarray):
        """
        adds a batch of prediction results to track, should be of shape
        (batch_size, num_labels)

        :param predictions: predicted scores from pipeline
        :param targets: target values - label column should be 1 if a label is positive
            0 otherwise
        """
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, predictions.shape[0])
        if targets.ndim == 1:
            targets = targets.reshape(1, targets.shape[0])

        if self._predictions is None:
            self._predictions = predictions
            self._targets = targets
        else:
            self._predictions = numpy.concatenate((self._predictions, predictions))
            self._targets = numpy.concatenate((self._targets, targets))

    def compute(self) -> Dict[str, float]:
        """
        computes per class and macro-averaged precision, recall, and f1 for multiple
        model sample predictions where targets may contain multiple labels

        :return: dictionary of per label and macro-average results for precision,
            recall, and f1
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            self._targets, self._predictions
        )

        # compile results into required str -> float dict
        results = {}
        for idx in range(self._predictions.shape[1]):
            label = self._id_to_label[idx] if self._id_to_label else str(idx)

            results[f"precision_{label}"] = precision[idx]
            results[f"recall_{label}"] = recall[idx]
            results[f"f1_{label}"] = f1[idx]

        # add macro averages and std to results
        results["precision_macro_average"] = precision.mean()
        results["recall_macro_average"] = recall.mean()
        results["f1_macro_average"] = f1.mean()

        results["precision_std"] = precision.std()
        results["recall_std"] = recall.std()
        results["f1_std"] = f1.std()

        return results
