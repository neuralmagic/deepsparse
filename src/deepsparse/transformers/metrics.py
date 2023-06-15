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


from typing import Dict, Optional

import numpy
from onnxruntime import InferenceSession
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import torch
from sklearn.metrics import precision_recall_fscore_support


__all__ = [
    "PrecisionRecallF1",
    "Perplexity",
]


class Perplexity:
    def __init__(
        self,
        session: InferenceSession,
        tokenizer: PreTrainedTokenizer,
        vocab_size: int,
        static_length: Optional[int] = None,
    ):
        """
        Given the onnxruntime session, compute the perplexity of the model
        on the given text input.
        Session will be in future swapped for the text generation pipeline.

        :param session: The onnxruntime session to use for inference
        :param tokenizer: The tokenizer to use for tokenizing the input text
        :param vocab_size: The size of the vocabulary for the model
        :param static_length: The static length of the input text to use
            for computing logits
        """

        self._session = session
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._static_length = static_length
        self._loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self.encoded_batches = None  # (batch_size, self._static_length)
        self.attention_masks = None  # (batch_size, self._static_length)

    def add_batch(
        self, input_text: str, batch_size: int = 16, add_start_token: bool = True
    ):
        """
        Converts input_text into data that can be eventually used to compute perplexity.

        :param input_text: The text to convert into data for computing perplexity
        :param batch_size: The batch size to use for tokenization
        :param add_start_token: Whether to add the start token to the input text
        """

        if add_start_token and self._static_length:
            # leave room for <BOS> token to be added:
            assert self._tokenizer.bos_token is not None, (
                "Input model must already have a BOS token "
                "if using add_start_token=True. Please use a "
                "different model, or set add_start_token=False"
            )
            max_tokenized_len = self._static_length - 1
        else:
            max_tokenized_len = self._static_length

        encodings = self._tokenizer(
            input_text,
            add_special_tokens=False,
            padding="max_length",
            max_length=max_tokenized_len,
            return_tensors="np",
            return_attention_mask=True,
        )

        encoded_texts = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attention_mask = attention_mask[start_index:end_index]

            if add_start_token:
                batch_size = encoded_batch.shape[0]
                # make tensor same shape as encoded_batch, but with <BOS> token
                bos_tokens = numpy.array([[self._tokenizer.bos_token_id]] * batch_size)
                encoded_batch = numpy.concatenate([bos_tokens, encoded_batch], axis=1)
                attention_mask = numpy.concatenate(
                    [numpy.ones(bos_tokens.shape, dtype=numpy.int64), attention_mask],
                    axis=1,
                )
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

    def compute(self) -> Dict[str, float]:
        """
        Given the data collected by add_batch() method,
        compute the perplexity of the model
        """
        perplexities = []
        logits_batch = self._session.run(
            ["logits"],
            dict(input_ids=self.encoded_batches, attention_mask=self.attention_masks),
        )[0]

        for idx, (logits, labels, attention_mask) in enumerate(
            zip(logits_batch, self.encoded_batches, self.attention_masks)
        ):
            # remove padding tokens
            logits = logits[: attention_mask.sum(), :]
            labels = labels[: attention_mask.sum()]
            attn_mask = attention_mask[: attention_mask.sum()]

            # shift logits and labels create the input and target for the loss function
            shift_logits = logits[:-1, :]
            shift_labels = labels[1:]
            shift_attention_mask_batch = attn_mask[1:]

            # compute perplexity for this batch
            perplexity_batch = torch.exp(
                (
                    self._loss_fct(
                        torch.tensor(shift_logits), torch.tensor(shift_labels)
                    )
                    * torch.tensor(shift_attention_mask_batch)
                ).sum()
                / torch.tensor(shift_attention_mask_batch).sum()
            )

            perplexities.append(perplexity_batch.item())
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
