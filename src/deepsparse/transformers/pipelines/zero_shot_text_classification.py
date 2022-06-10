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


from typing import List, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

import torch
from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "ZeroShotTextClassificationInput",
    "ZeroShotTextClassificationOutput",
    "ZeroShotTextClassificationPipeline",
]


class ZeroShotTextClassificationInput(BaseModel):
    """
    Schema for inputs to zero_shot_text_classification pipelines
    Each sequence and each candidate label must be paired and passed through
    the model, so the total number of forward passes is num_labels * num_sequences
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to "
        "zero_shot_text_classification task"
    )
    labels: Union[List[List[str]], List[str], str] = Field(
        description="The set of possible class labels to classify each "
        "sequence into. Can be a single label, a string of comma-separated "
        "labels, or a list of labels."
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
    mnli_text_classifier = Pipeline.create(
        task="mnli_text_classification",
        model_path="mnli_model_dir/",
    )
    ```

    example batch size 1, single sequence and three labels:
    ```python
    sequence_to_classify = "Who are you voting for in 2020?"
    candidate_labels = ["Europe", "public health", "politics"]
    mnli_text_classifier(sequence_to_classify, candidate_labels)
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
    :param hypothesis_template: formattable string that wraps the given labels
        Default is "This text is about {}"
    :param entailment_index: index of model output that represents mnli entailment
    :param contradiction_index: index of model output that represents mnli contradiction
    :param multi_class: true if class probablities are independent
    """

    def __init__(
        self,
        *,
        hypothesis_template: str = "This text is about {}",
        entailment_index: int = 0,
        contradiction_index: int = 2,
        multi_class: bool = False,
        **kwargs,
    ):
        self._hypothesis_template = hypothesis_template
        self._entailment_index = entailment_index
        self._contradiction_index = contradiction_index
        self._multi_class = multi_class

        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return ZeroShotTextClassificationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return ZeroShotTextClassificationOutput

    def _parse_labels(self, labels: Union[List[str], str]):
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

        num_sequences = 1 if isinstance(input.sequences, str) else len(input.sequences)
        if num_sequences != self._batch_size:
            raise ValueError(
                f"the number of sequences {num_sequences} must be equal to "
                f"the batch size the model was instantiated with {self._batch_size}"
            )

        return input

    def process_inputs(
        self, inputs: ZeroShotTextClassificationInput
    ) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            ZeroShotTextClassificationInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        sequences = inputs.sequences
        labels = inputs.labels

        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError(
                "You must include at least one label and at least one sequence."
            )

        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self._parse_labels(labels)

        if self._hypothesis_template.format(labels[0]) == self._hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be '
                    "formatted with the target labels. Make sure the passed template "
                    "includes formatting syntax such as {{}} where the label should go."
                ).format(self._hypothesis_template)
            )

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend(
                [
                    [sequence, self._hypothesis_template.format(label)]
                    for label in labels
                ]
            )

        tokens = self.tokenizer(
            sequence_pairs,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            # tokenize only_first so that hypothesis (label) is not truncated
            truncation=TruncationStrategy.ONLY_FIRST.value,
        )

        postprocessing_kwargs = dict(
            sequences=sequences,
            labels=labels,
        )

        return self.tokens_to_engine_input(tokens), postprocessing_kwargs

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        sequences = kwargs["sequences"]
        candidate_labels = kwargs["labels"]

        outputs = engine_outputs
        if isinstance(outputs, list):
            outputs = outputs[0]

        # Reshape sequences
        num_sequences = 1 if isinstance(sequences, str) else len(sequences)
        reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

        # Calculate scores
        entailment_contradiction_logits = reshaped_outputs[
            :, :, [self._entailment_index, self._contradiction_index]
        ]
        prob = torch.softmax(torch.tensor(entailment_contradiction_logits), dim=2)
        entailment_prob = prob[:, :, 0]
        if not self._multi_class:
            scores = torch.softmax(entailment_prob, dim=1)
        else:
            scores = entailment_prob
        scores = scores.numpy()

        # Hack: negate scores to perform reversed sort
        sorted_indexes = numpy.argsort(-1 * scores, axis=1)
        labels = [
            numpy.array(candidate_labels)[sorted_indexes[i]].tolist()
            for i in range(num_sequences)
        ]
        label_scores = numpy.take_along_axis(scores, sorted_indexes, axis=1).tolist()

        return self.output_schema(
            sequences=sequences,
            labels=labels,
            scores=label_scores,
        )

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        engine_inputs = numpy.array(engine_inputs)

        engine_outputs = []
        for batch_origin_index in range(0, engine_inputs.shape[1], self._batch_size):
            labelwise_inputs = engine_inputs[
                :, batch_origin_index : batch_origin_index + self._batch_size, :
            ]
            labelwise_inputs = [labelwise_input for labelwise_input in labelwise_inputs]
            engine_output = self.engine(labelwise_inputs)
            engine_outputs += engine_output

        engine_outputs = numpy.array(engine_outputs)
        return engine_outputs
