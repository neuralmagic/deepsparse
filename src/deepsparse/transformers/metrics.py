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

from scipy.special import log_softmax
from sklearn.metrics import precision_recall_fscore_support


__all__ = [
    "PrecisionRecallF1",
    "Perplexity",
]


class Perplexity:
    def __init__(self, accumulate: bool = False):
        """
        Class for computing perplexity.

        Each batch is processed via the "add_batches" method.
        At the end the data is reduced to a single perplexity
        metric via the "compute" method.

        Example:
        metric = Perplexity()
        for prediction, target in samples:
            metric.add_batch(prediction, target)
        perplexity_value = metric.compute()

        :param accumulate: If True, accumulate negative log-likelihood
            over samples. If False, perplexity is computed separately
            for each sampled and then averaged in the end.
        """
        self._predictions = None
        self._targets = None
        self._accumulate = accumulate
        if accumulate:
            self._neg_log_likelihood = 0.0
            self._number_tokens = 0
        else:
            self._perplexities = None

    def add_batch(self, predictions: numpy.ndarray, targets: numpy.ndarray):
        """
        Computes perplexity or negative log-likelihood for each batch
        (depending on accumulate argument)
        and track results.

        Tracks perplexity or negative log-likelihood since storing
        predictions may require a lot of memory.

        :param predictions: predicted scores.
            Accepted shapes:
              - [batch_size, sequence_length, vocab_size]
              - [sequence_length, vocab_size] (batch size = 1)
            Note: sequence length has to be uniform within a batch, but not all
              batches require the same sequence length
        :param targets: target values - index of correct vocabulary entry
        """

        if self._accumulate:
            # If accumulate is True, every token from the batch contributes
            # equally to the negative log-likelihood.
            # Thus, merge batch and sequence length dimensions and compute negative
            # log-likelihood for all tokens, and accumulate to total
            predictions = numpy.reshape(predictions, (-1, predictions.shape[-1]))
            targets = targets.flatten()

            # Compute negative log-likelihood and accumulate
            self._neg_log_likelihood += _cross_entropy(
                predictions, targets, reduction="sum"
            ).sum()

            # Track number of tokens processed
            self._number_tokens += predictions.shape[0]
        else:
            # If accumulate is False, compute perplexity for
            # each sample individually.
            # We assume that sequence length is uniform within a batch,
            # but may vary from batch to batch.

            # Create batch dimension if it doesn't exist
            if targets.ndim == 1:
                predictions = numpy.expand_dims(predictions, axis=0)
                targets = numpy.expand_dims(targets, axis=0)

            # Compute negative log-likelihoods for batch
            neg_log_likelihoods = _cross_entropy(predictions, targets)

            # Compute perplexities for batch
            perplexities = numpy.exp(neg_log_likelihoods)

            # Store perplexities
            if self._perplexities is None:
                self._perplexities = perplexities
            else:
                self._perplexities = numpy.concatenate(
                    (self._perplexities, perplexities)
                )

    def compute(self) -> Dict[str, Any]:
        """
        :return: A dictionary containing the final results.
        If accumulate is True, return single perplexity.
        Else, return a list of perplexities (one for each sample)
        and mean perplexity.
        """

        if self._accumulate:
            perplexity = numpy.exp(self._neg_log_likelihood / self._number_tokens)
            return {"perplexity": perplexity}
        else:
            return {
                "perplexities": self._perplexities,
                "mean_perplexity": numpy.mean(self._perplexities),
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


def _cross_entropy(
    predictions: numpy.ndarray,
    targets: numpy.ndarray,
    reduction: str = "mean",
) -> float:
    """
    Calculate the cross-entropy loss between predicted probabilities and target labels.

    Args:
        predictions (numpy.ndarray): Predicted logits.
        targets (nnumpy.ndarray): Target class labels.
        reduction (str, optional): Specifies the reduction method for the loss.
            - "mean" (default): Computes the mean loss over all samples.
            - "sum": Computes the sum of losses over all samples.

    Returns:
        float: The computed cross-entropy loss.
    """

    logp = log_softmax(predictions, axis=-1)
    neg_log_likelihoods = -1.0 * numpy.take_along_axis(
        logp, numpy.expand_dims(targets, axis=-1), axis=-1
    )
    neg_log_likelihoods = numpy.squeeze(neg_log_likelihoods, axis=-1)
    if reduction == "mean":
        neg_log_likelihoods = neg_log_likelihoods.mean(axis=-1)
    elif reduction == "sum":
        neg_log_likelihoods = neg_log_likelihoods.sum(axis=-1)

    return neg_log_likelihoods
