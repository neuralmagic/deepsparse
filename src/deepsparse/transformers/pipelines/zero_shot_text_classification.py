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

# postprocessing adapted from huggingface/transformers

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Pipeline implementation and pydantic models for zero-shot text classification
transformers tasks
"""


from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import List, Optional, Type, Union
from abc import ABC, abstractmethod

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.engine import Context
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.pipelines.nli_text_classification import NliTextClassificationPipeline

class ModelSchemes(str, Enum):
    """
    Enum containing all supported model schemes for zero shot text classification
    """

    nli = "nli"

    @classmethod
    def to_list(cls):
        return cls._value2member_map_


class ZeroShotTextClassificationOutput(BaseModel):
    """
    Schema for zero_shot_text_classification pipeline output. Values are in batch order
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to "
        "zero_shot_text_classification task"
    )
    labels: Union[List[List[str]], List[str]] = Field(
        description="The predicted labels in batch order"
    )
    scores: Union[List[List[float]], List[float]] = Field(
        description="The corresponding probability for each label in the batch"
    )


@Pipeline.register(
    task="zero_shot_text_classification",
    task_aliases=["zero-shot-text-classification", "zero_shot_text_classification"],
    default_model_path=(
        "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/"
        "mnli/pruned80_quant-none-vnni"
    ),
)
class ZeroShotTextClassificationPipeline(TransformersPipeline):
    """
    transformers zero shot text classification pipeline

    example dynamic labels:
    ```python
    zero_shot_text_classifier = Pipeline.create(
        task="zero_shot_text_classification",
        num_sequences=1,
        model_scheme="nli",
        model_config={"hypothesis_template": "This text is related to {}"},
        model_path="nli_model_dir/",
    )

    sequence_to_classify = "Who are you voting for in 2020?"
    candidate_labels = ["Europe", "public health", "politics"]
    zero_shot_text_classifier(sequence_to_classify, candidate_labels)
    >>> sequences=['Who are you voting for in 2020?']
        labels=[['politics', 'Europe', 'public health']]
        scores=[[0.7635, 0.1357, 0.1007]]
    ```

    example static labels:
    ```python
    zero_shot_text_classifier = Pipeline.create(
        task="zero_shot_text_classification",
        num_sequences=1,
        model_scheme="nli",
        model_path="nli_model_dir/",
        labels=["politics", "Europe", "public health"]
    )

    sequence_to_classify = "Who are you voting for in 2020?"
    zero_shot_text_classifier(sequence_to_classify)
    >>> sequences=['Who are you voting for in 2020?']
        labels=[['politics', 'Europe', 'public health']]
        scores=[[0.7635, 0.1357, 0.1007]]
    ```

    Note that labels must either be provided during pipeline instantiation via
    the constructor, at inference time, but not both.

    Note that if a hypothesis_template is provided at inference time, then it
    will override the value provided during model instantiation

    :param model_path: sparsezoo stub to a transformers model, an ONNX file, or
        (preferred) a directory containing a model.onnx, tokenizer config, and model
        config. If no tokenizer and/or model config(s) are found, then they will be
        loaded from huggingface transformers using the `default_model_name` key
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param default_model_name: huggingface transformers model name to use to
        load a tokenizer and model config when none are provided in the `model_path`.
        Default is "bert-base-uncased"
    :param model_scheme: training scheme used to train the model used for zero shot.
        Currently supported schemes are "nli"
    :param model_config: config object specific to the model_scheme of this model
        or a dict of config keyword arguments
    :param num_sequences: the number of sequences to handle per batch.
    :param labels: static list of labels to perform text classification with. Can
        also be provided at inference time
    :param context: context for engine. If None, then the engine will be initialized
        with 2 streams to make use of parallel inference of labels
    """

    def __new__(
        cls,
        model_path: str,
        model_scheme: str = ModelSchemes.nli.value,
        **kwargs,
    ):
        if model_scheme == ModelSchemes.nli:
            return NliTextClassificationPipeline(model_path, **kwargs)
        else:
            raise ValueError(
                f"Unknown model_scheme {model_scheme}. Currently supported model "
                f"schemes are {ModelSchemes.to_list()}"
            )
