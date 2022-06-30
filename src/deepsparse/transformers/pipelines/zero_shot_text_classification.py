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

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.engine import Context
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

    def __init__(
        self,
        *,
        model_scheme: str = ModelSchemes.nli.value,
        model_config: Optional[Union[NliTextClassificationConfig, dict]] = None,
        num_sequences: int = 1,
        labels: Optional[List] = None,
        context: Optional[Context] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        # Note: users should use num_sequences argument instead of batch_size
        if model_scheme not in ModelSchemes.to_list():
            raise ValueError(
                f"Unknown model_scheme {model_scheme}. Currently supported model "
                f"schemes are {ModelSchemes.to_list()}"
            )

        self._model_scheme = model_scheme
        self._config = self._parse_config(model_config)
        self._num_sequences = num_sequences
        self._labels = self._parse_labels(labels)
        self._thread_pool = None

        # if dynamic labels
        if self._labels is None:
            if context is None:
                # num_streams is arbitrarily chosen to be any value >= 2
                context = Context(num_cores=None, num_streams=2)
                kwargs.update({"context": context})

            self._thread_pool = ThreadPoolExecutor(
                max_workers=context.num_streams or 2,
                thread_name_prefix="deepsparse.pipelines.zero_shot_text_classifier",
            )

            if batch_size is not None and batch_size != num_sequences:
                raise ValueError(
                    f"This pipeline requires that batch_size {batch_size} match "
                    f"num_sequences {num_sequences} when no static labels are "
                    f"provided"
                )
            kwargs.update({"batch_size": num_sequences})

        # if static labels
        else:
            if batch_size is not None and batch_size != num_sequences * len(
                self._labels
            ):
                raise ValueError(
                    f"This pipeline requires that batch_size {batch_size} match "
                    f"num_sequences times the number labels "
                    f"{num_sequences * len(self._labels)} when static labels are "
                    f"provided"
                )
            kwargs.update({"batch_size": num_sequences * len(self._labels)})

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

    def _parse_labels(self, labels: Union[None, List[str], str]) -> List[str]:
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

        # check for absent labels
        if inputs.labels is None and self._labels is None:
            raise ValueError(
                "You must provide either static labels during pipeline creation or "
                "dynamic labels at inference time"
            )

        # check for conflicting labels
        if inputs.labels is not None and self._labels is not None:
            raise ValueError(
                "Found both static labels and dynamic labels at inference time. You "
                "must provide only one"
            )

        # check for incorrect number of sequences
        num_sequences = (
            1 if isinstance(inputs.sequences, str) else len(inputs.sequences)
        )
        if num_sequences != self._num_sequences:
            raise ValueError(
                f"number of sequences {num_sequences} must match the number of "
                f"sequences the pipeline was instantiated with {self._num_sequences}"
            )

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

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        current_seq_len = (
            ZeroShotTextClassificationPipeline._get_current_sequence_length(
                input_schema
            )
        )

        for pipeline in pipelines:
            if pipeline.sequence_length > current_seq_len:
                return pipeline
        return pipelines[-1]

    @staticmethod
    def _get_current_sequence_length(input_schema: BaseModel) -> int:
        """
        Helper function to get max sequence length in provided sequences input

        :param input_schema: input to pipeline
        :return: max sequence length in input_schema
        """
        if isinstance(input_schema.sequences, str):
            current_seq_len = len(input_schema.sequences.split())
        elif isinstance(input_schema.sequences, list):
            current_seq_len = float("-inf")
            for _input in input_schema.sequences:
                if isinstance(_input, str):
                    current_seq_len = max(len(_input.split()), current_seq_len)
                elif isinstance(_input, list):
                    current_seq_len = max(
                        *(len(__input.split()) for __input in _input), current_seq_len
                    )
        else:
            raise ValueError(
                "Expected a str or List[str] or List[List[str]] as input but got "
                f"{type(input_schema.sequences)}"
            )

        return current_seq_len
