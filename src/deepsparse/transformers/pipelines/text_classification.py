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
Pipeline implementation and pydantic models for text classification transformers
tasks
"""


from typing import List, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "TextClassificationInput",
    "TextClassificationOutput",
    "TextClassificationPipeline",
]


class TextClassificationInput(BaseModel):
    """
    Schema for inputs to text_classification pipelines
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to"
        "text_classification task"
    )


class TextClassificationOutput(BaseModel):
    """
    Schema for text_classification pipeline output. Values are in batch order
    """

    labels: List[str] = Field(description="The predicted labels in batch order")
    scores: List[float] = Field(
        description="The corresponding probability for each label in the batch"
    )


@Pipeline.register(
    task="text_classification",
    task_aliases=["glue", "sentiment_analysis"],
)
class TextClassificationPipeline(TransformersPipeline):
    """
    transformers text classification pipeline

    example instantiation:
    ```python
    text_classifier = Pipeline.create(
        task="text_classification",
        model_path="text_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```

    example batch size 1, single text inputs (ie sentiment analysis):
    ```python
    sentiment = text_classifier("the food tastes great")
    sentiment = text_classifier(["the food tastes great"])
    sentiment = text_classifier([["the food tastes great"]])
    ```

    example batch size 1, multi text input (ie QQP like tasks):
    ```python
    prediction = text_classifier([["how is the food?", "what is the food?"]])
    ```

    example batch size n, single text inputs:
    ```python
    sentiments = text_classifier(["the food tastes great", "the food tastes bad"])
    sentiments = text_classifier([["the food tastes great"], ["the food tastes bad"]])
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
        Default is 'bert-base-uncased'
    """

    @property
    def input_model(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return TextClassificationInput

    @property
    def output_model(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return TextClassificationOutput

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        """
        :param args: ordered arguments to pipeline, only an input_model object
            is supported as an arg for this function
        :param kwargs: keyword arguments to pipeline
        :return: pipeline arguments parsed into the given `input_model`
            schema if necessary. If an instance of the `input_model` is provided
            it will be returned
        """
        if args and kwargs:
            raise ValueError(
                f"{self.__class__} only support args OR kwargs. Found "
                f" {len(args)} args and {len(kwargs)} kwargs"
            )

        if args:
            if len(args) == 1:
                # passed input_model schema directly
                if isinstance(args[0], self.input_model):
                    return args[0]
                return self.input_model(sequences=args[0])
            else:
                return self.input_model(sequences=args)

        return self.input_model(**kwargs)

    def process_inputs(self, inputs: TextClassificationInput) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            TextClassificationInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        tokens = self.tokenizer(
            inputs.sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
        )
        return self.tokens_to_engine_input(tokens)

    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray]) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_model`
            format of this pipeline
        """
        outputs = engine_outputs
        if isinstance(outputs, list):
            outputs = outputs[0]

        scores = (
            1.0 / (1.0 + numpy.exp(-outputs))
            if self.config.num_labels == 1
            else numpy.exp(outputs) / numpy.exp(outputs).sum(-1, keepdims=True)
        )

        labels = []
        label_scores = []

        for score in scores:
            labels.append(self.config.id2label[score.argmax()])
            label_scores.append(score.max().item())

        return self.output_model(
            labels=labels,
            scores=label_scores,
        )
