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


from enum import Enum
from typing import List, Optional, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.pipelines.nli_text_classification import (
    NliTextClassificationConfig,
    NliTextClassificationInput,
    nli_engine_forward,
    process_nli_engine_outputs,
    process_nli_inputs,
)


__all__ = [
    "ZeroShotTextClassificationOutput",
    "ZeroShotTextClassificationPipeline",
]


class ModelSchemes(Enum):
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
    transformers zero-shot zero shot text classification pipeline

    example instantiation:
    ```python
    zero_shot_text_classifier = Pipeline.create(
        task="zero_shot_text_classification",
        model_scheme="nli",
        model_scheme_config={"multi_class": False},
        model_path="nli_model_dir/",
    )
    ```

    example batch size 1, single sequence and three labels:
    ```python
    sequence_to_classify = "Who are you voting for in 2020?"
    candidate_labels = ["Europe", "public health", "politics"]
    zero_shot_text_classifier(sequence_to_classify, candidate_labels)
    >>> {'sequence': 'Who are you voting for in 2020?',
        'labels': ['politics', 'Europe', 'public health'],
        'scores': [0.9676, 0.0195, 0.0128]}
    ```

    :param model_path: sparsezoo stub to a transformers model, an ONNX file, or
        (preferred) a directory containing a model.onnx, tokenizer config, and model
        config. If no tokenizer and/or model config(s) are found, then they will be
        loaded from huggingface transformers using the `default_model_name` key
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: static batch size to use for inference. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param sequence_length: sequence length to compile model and tokenizer for.
        Default is 128
    :param default_model_name: huggingface transformers model name to use to
        load a tokenizer and model config when none are provided in the `model_path`.
        Default is "bert-base-uncased"
    :param model_scheme: training scheme used to train the model used for zero shot.
        Currently supported schemes are "nli"
    :param model_scheme_config: Config object or a dict of config keyword arguments
    """

    def __init__(
        self,
        *,
        model_scheme: str = ModelSchemes.nli.value,
        model_scheme_config: Optional[Union[NliTextClassificationConfig, dict]] = None,
        **kwargs,
    ):
        if model_scheme not in ModelSchemes.to_list():
            raise ValueError(
                f"Unknown model_scheme {model_scheme}. Currently supported model "
                f"schemes are {ModelSchemes.to_list()}"
            )

        self._model_scheme = model_scheme
        self._config = self._parse_config(model_scheme_config)

        super().__init__(**kwargs)

    def _parse_config(
        self, model_scheme_config: Optional[Union[NliTextClassificationConfig, dict]]
    ) -> Type[BaseModel]:
        """
        :param model_scheme_config: optional config arguments specified by user
        :return: instance of config pydantic model for this pipeline's model scheme
        """
        model_scheme_config = model_scheme_config if model_scheme_config else {}

        if self._model_scheme == ModelSchemes.nli.value:
            config_schema = NliTextClassificationConfig
        else:
            raise Exception(f"Unknown model_scheme {self._model_scheme}")

        if isinstance(model_scheme_config, config_schema):
            return model_scheme_config

        elif isinstance(model_scheme_config, dict):
            return config_schema(**model_scheme_config)

        else:
            raise ValueError(
                f"pipeline {self.__class__} only supports either only a "
                f"{config_schema} object a dict of keywords used to construct "
                f"one. Found {model_scheme_config} instead"
            )

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        if self._model_scheme == ModelSchemes.nli.value:
            return NliTextClassificationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return ZeroShotTextClassificationOutput

    def _parse_labels(self, labels: Union[List[str], str]) -> List[str]:
        """
        If given a string of comma separated labels, parses values into a list

        :param labels: A string of comma separated labels or a list of labels
        :return: a list of labels, parsed if originally in string form
        """
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        """
        :param args: ordered arguments to pipeline, only an input_schema object
            is supported as an arg for this function
        :param kwargs: keyword arguments to pipeline
        :return: pipeline arguments parsed into the given `input_schema`
            schema if necessary. If an instance of the `input_schema` is provided
            it will be returned
        """
        if args and kwargs:
            raise ValueError(
                f"{self.__class__} only support args OR kwargs. Found "
                f" {len(args)} args and {len(kwargs)} kwargs"
            )

        if args:
            if len(args) == 1 and isinstance(args[0], self.input_schema):
                input = args[0]
            else:
                input = self.input_schema(*args)
        else:
            input = self.input_schema(**kwargs)

        return input

    def process_inputs(
        self, inputs: Union[NliTextClassificationInput]
    ) -> List[numpy.ndarray]:
        """
        Decides how to process inputs based on the _model_scheme of the pipeline

        :param inputs: inputs to the pipeline.
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        if self._model_scheme == ModelSchemes.nli.value:
            return process_nli_inputs(self, inputs, self._config)

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        Decides how to process engine outputs based on the _model_scheme of the pipeline

        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        if self._model_scheme == ModelSchemes.nli.value:
            return process_nli_engine_outputs(
                self, engine_outputs, self._config, **kwargs
            )

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        Decides how to carry out about engine forward based on the _model_scheme of
        the pipeline.

        :param engine_inputs: list of numpy inputs to Pipeline engine forward
            pass
        :return: result of forward pass to Pipeline engine
        """
        if self._model_scheme == ModelSchemes.nli.value:
            return nli_engine_forward(self, engine_inputs)
