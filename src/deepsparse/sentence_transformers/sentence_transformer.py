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
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm.autonotebook import trange
from transformers.onnx.utils import get_preprocessor

import torch
from optimum.deepsparse import DeepSparseModelForFeatureExtraction


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "zeroshot/bge-small-en-v1.5-quant"


class SentenceTransformer:
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        export: bool = False,
        max_seq_length: int = 512,
    ):

        self.model_name_or_path = model_name_or_path
        self.model = DeepSparseModelForFeatureExtraction.from_pretrained(
            model_name_or_path, export=export
        )
        self.tokenizer = get_preprocessor(model_name_or_path)

        self._max_seq_length = max_seq_length
        self._batch_size = 1

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 1,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:

        # TODO: support executing with batch size > 1
        batch_size = 1

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            model_inputs = self.tokenize(sentences_batch)
            model_output = self.model(**model_inputs)

            out_features = {}
            out_features["sentence_embedding"] = self.mean_pooling(
                model_output, model_inputs["attention_mask"]
            )

            if output_value == "token_embeddings":
                embeddings = []
                for token_emb, attention in zip(
                    out_features[output_value], out_features["attention_mask"]
                ):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0 : last_mask_id + 1])
            elif output_value is None:
                # Return all outputs
                embeddings = []
                for sent_idx in range(len(out_features["sentence_embedding"])):
                    row = {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
            else:
                # Sentence embeddings
                embeddings = out_features[output_value]
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def get_max_seq_length(self) -> int:
        """
        Returns the maximal sequence length for input the model accepts.
        Longer inputs will be truncated
        """
        return self._max_seq_length

    def _text_length(self, text: Union[List[int], List[List[int]]]) -> int:
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes the texts
        """
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean pooling of token embeddings weighted by attention mask.
        Args:
            model_output (torch.Tensor): The model's output tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
        Returns:
            torch.Tensor: Mean-pooled embeddings.
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
