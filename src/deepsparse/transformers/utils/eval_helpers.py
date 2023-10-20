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

from typing import List, Mapping, Union

import numpy
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from datasets import load_dataset


CONCATENATED_DATSETS = ["wikitext2", "c4"]


def process_concatenated_datasets(
    dataset_name: str,
    model_path: str,
    max_sequence_length: int,
    kwargs: Mapping,
) -> list:
    """
    Concatenate text datasets and split them into chunks text that, after
    tokenization, have size "max_sequence_length" tokens.

    Args:
        dataset_name (str): The name of the dataset to process.
            Options: "wikitext2" or "c4".
        model_path (str): The path to a pretrained transformer model for tokenization.
        max_sequence_length (int): The maximum number of tokens in each sequence.
        kwargs (mapping): Additional keyword arguments.
            - eos (str, optional): The end-of-sentence token.
                Default is "\n\n" for wikitext2 and "" for c4.
            - bos (str, optional): The beginning-of-sentence token.
                Default is "".
            - raw_samples (int, optional): The number of raw samples to use.
                Default is None.
            - data_file (int, optional): The index of the data file to use for dataset.
                Not used in wikitext2. Default is 0 for c4.
            - max_text_length (int, optional): The maximum length of text to consider.
    Returns:
        list: A list of text sequences.

    Raises:
        ValueError: If an invalid dataset_name is provided.
    """

    if dataset_name not in CONCATENATED_DATSETS:
        raise KeyError(
            f"dataset {dataset_name} not supported for concatenated processing, "
            f"available datasets are {list(CONCATENATED_DATSETS.keys())}"
        )

    if dataset_name == "wikitext2":
        eos = kwargs.get("eos", "\n\n")
        bos = kwargs.get("bos", "")

        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        raw_text = raw_dataset["text"]
    elif dataset_name == "c4":
        eos = kwargs.get("eos", "<|endoftext|>")
        bos = kwargs.get("bos", "")
        raw_samples = kwargs.get("raw_samples", None)
        data_file = kwargs.get("data_file", 0)
        if data_file is not None:
            raw_dataset = load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={
                    "validation": f"en/c4-validation.{data_file:05d}-of-00008.json.gz"
                },
                split="validation",
            )
        else:
            raw_dataset = load_dataset(
                "allenai/c4",
                "allenai--c4",
                split="validation",
            )
        if raw_samples is not None:
            raw_dataset = raw_dataset[:raw_samples]
        raw_text = raw_dataset["text"]

    # Dataset is split into sections that contain "max_sequence_length" tokens.
    # To split the dataset, first tokenize text
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return _split_text_by_tokens(
        raw_text,
        eos,
        bos,
        tokenizer,
        max_sequence_length,
        kwargs.get("max_text_length", None),
    )


def _split_text_by_tokens(
    text: List[str],
    eos: str,
    bos: str,
    tokenizer: PreTrainedTokenizerFast,
    sequence_length: int,
    max_text_length: Union[None, int],
) -> List[str]:
    """
    Tokenizes and splits a list of concatenated text samples into
    sections of specified maximum token length.

    Args:
        text (List[str]): List of concatenated text samples to be tokenized and split.
        eos (str): The end-of-sentence token.
        bos (str): The beginning-of-sentence token.
        tokenizer (PreTrainedTokenizerFast): Tokenizer for tokenizing the text.
        sequence_length (int): The maximum number of tokens in each section.
        max_text_length (Union[None, int]): The maximum length of text to consider.
            - If None, the entire text is tokenized and split.
            - If -1, each sample is tokenized separately.
            - If a positive integer, the text is split into sections of this
                length before tokenization.

    Returns:
        List[str]: A list of sections where each section contains a
            maximum of "sequence_length" tokens.
    """

    text = [bos + sample + eos for sample in text]

    if max_text_length is None:
        text = "".join(text)
        input_tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    elif max_text_length == -1:  # per sample tokenization
        input_tokens = []
        for slice in text:
            input_tokens.append(tokenizer(slice, return_tensors="np")["input_ids"][0])
        input_tokens = numpy.concatenate(input_tokens)
    else:
        text = "".join(text)
        text_slices = len(text) // max_text_length
        sliced_text = [
            text[i * max_text_length : (i + 1) * max_text_length]
            for i in range(text_slices)
        ]
        sliced_text.append(text[text_slices * max_text_length :])
        input_tokens = []
        for slice in sliced_text:
            input_tokens.append(tokenizer(slice, return_tensors="np")["input_ids"][0])
        input_tokens = numpy.concatenate(input_tokens)

    # Then split the tokenized text into sections of size "max_sequence_length" and
    # decode each section back into text format
    split_text = []
    for i in range(len(input_tokens) // sequence_length):
        start = i * sequence_length
        end = (i + 1) * sequence_length
        split_text.append(
            tokenizer.decode(
                input_tokens[start:end],
                clean_up_tokenization_spaces=False,
            )
        )

    # Handle any leftover tokens
    if (i + 1) * sequence_length < len(input_tokens):
        start = (i + 1) * sequence_length
        end = len(input_tokens)
        split_text.append(
            tokenizer.decode(
                input_tokens[start:end],
                clean_up_tokenization_spaces=False,
            )
        )

    return split_text
