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
Bouncer class and pydantic models for zero-shot text classification transformers tasks

ZeroShotTextClassificationPipeline acts as a bouncer class, where instead of returning
an instance of itself, the ZeroShotTextClassificationPipeline returns an instance of
a pipeline which implements zero shot text classification. Which type of pipeline
is returned is specified by the model_scheme argument.

example
```
zero_shot_text_classification_pipeline = ZeroShotTextClassificationPipeline(
    model_scheme="mnli"
)
assert isinstance(zero_shot_text_classification_pipeline,
                  MnliTextClassificationPipeline
)
```

Pipeline implementations inherit from ZeroShotTextClassificationPipelineBase and
their inputs inherit from ZeroShotTextClassificationInputBase to ensure some
standards across implementations.
"""


from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


if TYPE_CHECKING:
    from deepsparse.transformers.pipelines.mnli_text_classification import (  # noqa: F401, E501
        MnliTextClassificationConfig,
    )

__all__ = [
    "ZeroShotTextClassificationPipeline",
    "ZeroShotTextClassificationInputBase",
    "ZeroShotTextClassificationOutput",
    "ZeroShotTextClassificationPipelineBase",
    "ModelSchemes",
]


class ModelSchemes(str, Enum):
    """
    Enum containing all supported model schemes for zero shot text classification
    """

    mnli = "mnli"

    @classmethod
    def to_list(cls) -> List[str]:
        return cls._value2member_map_


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
    Transformers zero shot text classification pipeline. This pipeline allows for
    text classification using models which were trained on datasets not originally
    meant for this task.

    This class upon construction returns an instance of a child Pipeline which
    inherits from ZeroShotTextClassificationPipelineBase. Which type of Pipeline
    is returned depends on the value of the passed model_scheme argument.

    example dynamic labels:
    ```python
    zero_shot_text_classifier = Pipeline.create(
        task="zero_shot_text_classification",
        model_scheme="mnli",
        model_config={"hypothesis_template": "This text is related to {}"},
        model_path="mnli_model_dir/",
    )

    sequence_to_classify = "Who are you voting for in 2020?"
    candidate_labels = ["Europe", "public health", "politics"]
    zero_shot_text_classifier(sequences=sequence_to_classify, labels=candidate_labels)
    >>> ZeroShotTextClassificationOutput(
        sequences='Who are you voting for in 2020?',
        labels=['politics', 'public health', 'Europe'],
        scores=[0.9073666334152222, 0.046810582280159, 0.04582275450229645])
    ```

    example static labels:
    ```python
    zero_shot_text_classifier = Pipeline.create(
        task="zero_shot_text_classification",
        model_scheme="mnli",
        model_config={"hypothesis_template": "This text is related to {}"},
        model_path="mnli_model_dir/",
        labels=["politics", "Europe", "public health"]
    )

    sequence_to_classify = "Who are you voting for in 2020?"
    zero_shot_text_classifier(sequences=sequence_to_classify)
    >>> ZeroShotTextClassificationOutput(
        sequences='Who are you voting for in 2020?',
        labels=['politics', 'public health', 'Europe'],
        scores=[0.9073666334152222, 0.046810582280159, 0.04582275450229645])
    ```

    Note that labels must either be provided during pipeline instantiation via
    the constructor, at inference time, but not both.

    Note that if a hypothesis_template is provided at inference time, then it
    will override the value provided during model instantiation

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: batch size must divide sequences * labels, regardless of
        whether using dynamic or static labels. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param default_model_name: huggingface transformers model name to use to
        load a tokenizer and model config when none are provided in the `model_path`.
        Default is "bert-base-uncased"
    :param model_scheme: training scheme used to train the model used for zero shot.
        Default is "mnli"
    :param model_config: config object specific to the model_scheme of this model
        or a dict of config keyword arguments
    :param labels: static list of labels to perform text classification with. Can
        also be provided at inference time
    :param context: context for engine. If None, then the engine will be initialized
        with 2 streams to make use of parallel inference of labels
    """

    def __new__(
        cls,
        model_path: str,
        model_scheme: str = ModelSchemes.mnli.value,
        **kwargs,
    ):
        pipeline = None
        if model_scheme == ModelSchemes.mnli:
            from deepsparse.transformers.pipelines.mnli_text_classification import (
                MnliTextClassificationPipeline,
            )

            pipeline = MnliTextClassificationPipeline(model_path, **kwargs)
        else:
            raise ValueError(
                f"Unknown model_scheme {model_scheme}. Currently supported model "
                f"schemes are {ModelSchemes.to_list()}"
            )
        pipeline.task = cls.task
        return pipeline


class ZeroShotTextClassificationInputBase(BaseModel):
    """
    Schema for inputs to zero_shot_text_classification pipelines
    Each sequence and each candidate label must be paired and passed through
    the model, so the total number of forward passes is num_labels * num_sequences
    """

    sequences: Union[List[str], str] = Field(
        description="A string or List of strings representing input to "
        "zero_shot_text_classification task"
    )


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


class ZeroShotTextClassificationPipelineBase(TransformersPipeline):
    """
    Base class for implementing zero shot text classification. Implementations of
    zero shot text classification inherit from this class.

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
    :param context: context for engine. If None, then the engine will be initialized
        with 2 streams to make use of parallel inference of labels
    """

    @property
    def output_schema(self) -> Type[ZeroShotTextClassificationOutput]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return ZeroShotTextClassificationOutput

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
            if len(args) == 1:
                # passed input_schema schema directly
                if isinstance(args[0], self.input_schema):
                    return args[0]
                return self.input_schema(sequences=args[0])
            else:
                return self.input_schema(sequences=args)

        return self.input_schema(**kwargs)

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        tokenizer = pipelines[0].tokenizer
        tokens = tokenizer(
            input_schema.sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=False,
            truncation=False,
        )
        input_seq_len = max(map(len, tokens["input_ids"]))
        return TransformersPipeline.select_bucket_by_seq_len(input_seq_len, pipelines)

    def parse_labels(self, labels: Union[None, List[str], str]) -> List[str]:
        """
        If given a string of comma separated labels, parses values into a list

        :param labels: A string of comma separated labels or a list of labels
        :return: a list of labels, parsed if originally in string form
        """
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def parse_config(
        self, model_scheme_config: Optional[Union["MnliTextClassificationConfig", Dict]]
    ) -> Type["MnliTextClassificationConfig"]:
        """
        :param model_scheme_config: optional config arguments specified by user
        :return: instance of config pydantic model for this pipeline's model scheme
        """
        model_scheme_config = model_scheme_config if model_scheme_config else {}

        if isinstance(model_scheme_config, self.config_schema):
            return model_scheme_config

        elif isinstance(model_scheme_config, dict):
            return self.config_schema(**model_scheme_config)

        else:
            raise ValueError(
                f"pipeline {self.__class__} only supports either only a "
                f"{self.config_schema} object a dict of keywords used to construct "
                f"one. Found {model_scheme_config} instead"
            )
