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
        self, input_text: str, batch_size: int = 16, add_start_token: bool = True
    ):
        """
        Converts input_text into data that can be eventually used to compute perplexity.
        Note: BOS token means "Begging of Sentence" token, which as
              the same as SOS token "Start of Sentence" token.

        :param input_text: The text to convert into data for computing perplexity
        :param batch_size: The batch size to split the input text into
         non-overlapping batches
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

        # split input_text into non-overlapping batches of `batch_size`
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
        for idx, (encoded_batch, attention_mask) in enumerate(
            zip(self.encoded_batches, self.attention_masks)
        ):
            batch_logits = []
            tokens = encoded_batch.tolist()[: attention_mask.sum()]
            batch_sequences = [
                self._tokenizer.decode(tokens[:i], skip_special_tokens=True)
                for i in range(1, len(tokens) + 1)
            ]
            for sequence in batch_sequences:
                # cannot do it in batch, we need to run
                # p(x_i | x_1, ..., x_{i-1}) for each i
                out = self._pipeline(sequence=sequence, return_logits=True)
                batch_logits.append(out.logits)

            logits = numpy.concatenate(batch_logits, axis=1)[0]

            # extract only the meaningful info from the
            # data that assumes static length
            labels = encoded_batch[: attention_mask.sum()]
            attention_mask = attention_mask[: attention_mask.sum()]

            # shift logits and labels create the input and target for the loss function
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            shift_attention_mask_batch = attention_mask[1:]

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
