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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.autonotebook import trange
from transformers.onnx.utils import get_preprocessor

import torch
from optimum.deepsparse import DeepSparseModelForFeatureExtraction


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "zeroshot/bge-small-en-v1.5-quant"


class DeepSparseSentenceTransformer:
    """
    Loads or creates a SentenceTransformer-compatible model that can be used to map
    text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from
        that path. If it is not a path, it first tries to download and export a model
        from a HuggingFace models repository with that name.
    :param export: To load a PyTorch checkpoint and convert it to the DeepSparse
        format on-the-fly, you can set `export=True` when loading your model.
    :param max_seq_length: Sets a limit on the maxmimum sequence length allowed,
        this should be set to 512 for most models. Any text that exceeds this
        token length will be truncated.
    :param use_auth_token: HuggingFace authentication token to download private models.
    :param buckets: Create static buckets less than max_seq_length automaticly if True,
        manually specified if a List of lengths are passed in, or fully dynamic if False
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        export: bool = False,
        max_seq_length: int = 512,
        use_auth_token: Union[bool, str, None] = None,
        buckets: Union[bool, List[int]] = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = get_preprocessor(model_name_or_path)
        self._max_seq_length = max_seq_length
        # TODO: support faster bulk execution with batch size > 1
        self._static_batch_size = 1

        self.dyn_model = DeepSparseModelForFeatureExtraction.from_pretrained(
            model_name_or_path,
            export=export,
            use_auth_token=use_auth_token,
        )
        self.dyn_model.reshape(input_shapes="[-1,-1]")
        self.dyn_model.compile(batch_size=0)

        if buckets:
            # Initialize a model for each bucket
            self.buckets = (
                buckets
                if isinstance(buckets, list)
                else [int(self._max_seq_length / 4 * i) for i in range(1, 5)]
            )
            self.models = {}
            for bucket in self.buckets:
                self.models[
                    bucket
                ] = DeepSparseModelForFeatureExtraction.from_pretrained(
                    model_name_or_path,
                    export=export,
                    use_auth_token=use_auth_token,
                )
                self.models[bucket].reshape(
                    input_shapes=f"[{self._static_batch_size},{bucket}]"
                )
                self.models[bucket].compile(batch_size=self._static_batch_size)
        else:
            self.buckets = None
            self.models = None

    def _select_bucket(self, seq_length: int) -> int:
        """
        Selects the appropriate model based on the input sequence length.
        """
        for bucket in self.buckets:
            if seq_length <= bucket:
                return bucket
        # default to the maximum if seq_length exceeds all buckets
        return self._max_seq_length

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
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings.
            Can be set to token_embeddings to get wordpiece token embeddings. Set to
            None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors.
            Else, it is a list of PyTorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return.
            Overwrites any setting from convert_to_numpy
        :param normalize_embeddings: If set to true, returned vectors will have
            length 1. In that case, the faster dot-product (util.dot_score)
            instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor,
           a stacked tensor is returned. If convert_to_numpy, a numpy matrix
           is returned.
        """

        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (
                logging.INFO,
                logging.DEBUG,
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

            if self.buckets and batch_size == 1:
                # Use bucketing for batch size 1
                # Select the model based on the bucketing logic
                # TODO: tokenize ahead of time and simply add padding
                seq_length = len(model_inputs[0])
                selected_bucket = self._select_bucket(seq_length)

                # Tokenize using the selected bucket size
                model_inputs = self.tokenize(
                    sentences_batch, target_length=selected_bucket
                )
                model = self.models[selected_bucket]
            else:
                # Use dynamic shape
                model = self.dyn_model

            # Run the inference
            model_output = model(**model_inputs)

            out_features = {}
            out_features["sentence_embedding"] = self.mean_pooling(
                model_output, model_inputs["attention_mask"]
            )

            embeddings = []
            if output_value == "token_embeddings":
                for token_emb, attention in zip(
                    out_features[output_value], out_features["attention_mask"]
                ):
                    # Apply the attention mask to remove embeddings for padding tokens
                    # Count non-zero values in the attention mask
                    actual_tokens_count = attention.sum().item()
                    # Slice the embeddings using this count
                    embeddings.append(token_emb[:actual_tokens_count])
            elif output_value is None:
                # Return all outputs
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

    def tokenize(
        self,
        texts: Union[List[str], List[Dict], List[Tuple[str, str]]],
        target_length: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Tokenizes the texts
        """
        if target_length:
            # Static length:
            # Make sure to pad and truncate tokens to the specified length
            return self.tokenizer(
                texts,
                max_length=target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            # Dynamic length:
            # Pad only to the maximum sequence length of the batch,
            # and truncate to _max_seq_length if needed
            return self.tokenizer(
                texts,
                max_length=self._max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

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


# for backwards compatibility
SentenceTransformer = DeepSparseSentenceTransformer
