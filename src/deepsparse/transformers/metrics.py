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


from typing import Any, Dict, List, Optional

import numpy
from tqdm import tqdm
from transformers import AutoTokenizer

import torch
from deepsparse import Pipeline
from deepsparse.transformers.pipelines.text_generation import TextGenerationPipeline
from sklearn.metrics import precision_recall_fscore_support


__all__ = [
    "PrecisionRecallF1",
    "Perplexity",
]


class Perplexity:
    def __init__(
        self,
        pipeline: Pipeline,
    ):
        """
        Given the pipeline, compute the perplexity of the model
        on the given text input.

        Code adapted from:
        https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py # noqa: E501

        :param pipeline: The pipeline to use for text generation
        """
        if not isinstance(pipeline, TextGenerationPipeline):
            raise ValueError(
                "Perplexity can only be computed for text generation pipelines"
            )
        self._pipeline = pipeline
        self._tokenizer = AutoTokenizer.from_pretrained(pipeline.model_path)
        self._vocab_size = pipeline.config.vocab_size
        self._static_length = pipeline.sequence_length
        self._loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self.encoded_batches = None  # (batch_size, self._static_length)
        self.attention_masks = None  # (batch_size, self._static_length)

    def add_batch(
        self, predictions: List[str], batch_size: int = 16, add_start_token: bool = True
    ):
        """
        Converts input_text into data that can be eventually used to compute perplexity.
        Note: BOS token means "Beginning of Sentence" token, which as
              the same as SOS token "Start of Sentence" token.

        :param predictions: The predictions to compute perplexity on
        :param batch_size: The batch size to split the input text into
         non-overlapping batches
        :param add_start_token: Whether to add the start token to the input text
        """

        self._tokenizer.pad_token = self._tokenizer.eos_token
        # tokenize list of strings

        encodings = self._tokenizer(
            predictions,
            return_attention_mask=True,
            max_length=self._static_length,
            truncation=True,
            padding="max_length",
        )
        # undo what this tokenizer does

        encoded_texts = encodings["input_ids"]
        attention_masks = encodings["attention_mask"]

        # split input_text into non-overlapping batches of `batch_size`
        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attention_mask = attention_masks[start_index:end_index]

            # save batches in the class object's state
            self.encoded_batches = (
                encoded_batch
                if self.encoded_batches is None
                else numpy.concatenate([self.encoded_batches, encoded_batch], axis=0)
            )
            self.attention_masks = (
                attention_mask
                if self.attention_masks is None
                else numpy.concatenate([self.attention_masks, attention_mask], axis=0)
            )

    def compute(self) -> Dict[str, Any]:
        """
        Given the data collected by add_batch() method,
        compute the perplexity of the model
        """
        perplexities = []

        """
        Because we are not able to run batched inference
        on the pipeline, we need to run inference on each
        sequence in the batch individually.
        In the future, once the batch support is ready,
        we could simply run in the pipeline
        ```
        out = self._pipeline(sequence=func(self.encoded_batches))
        ```
        """
        out = self._pipeline(
            input_ids_and_masks=(
                numpy.stack(self.encoded_batches),
                numpy.stack(self.attention_masks),
            ),
            return_logits=True,
        )
        logits = out.logits

        labels = self.encoded_batches

        # shift logits and labels create the input and target for the loss function
        shift_logits = logits[:, :-1, :]
        shift_labels = numpy.stack(labels)[
            :, 1:
        ]  # (batch_size - 1, self._static_length)
        shift_attention_mask_batch = numpy.stack(self.attention_masks)[
            :, 1:
        ]  # (batch_size - 1, self._static_length)

        # compute perplexity for this batch
        perplexity_batch = torch.exp(
            (
                self._loss_fct(
                    torch.tensor(shift_logits.transpose(0, 2, 1)),
                    torch.tensor(shift_labels),
                )
                * torch.tensor(shift_attention_mask_batch)
            ).sum(1)
            / torch.tensor(shift_attention_mask_batch).sum(1)
        )

        return {
            "perplexities": perplexity_batch.numpy().tolist(),
            "mean_perplexity": perplexity_batch.mean().item(),
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
